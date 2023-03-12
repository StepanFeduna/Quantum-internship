import rasterio
from rasterio.plot import reshape_as_image
import rasterio.mask
from rasterio.features import rasterize

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models


def poly_from_utm(polygon, transform):
    poly_pts = []

    # make a polygon from multipolygon
    poly = cascaded_union(polygon)
    for i in np.array(poly.exterior.coords):
        # transfrom polygon to image crs, using raster meta
        poly_pts.append(~transform * tuple(i))

    # make a shapely Polygon object
    new_poly = Polygon(poly_pts)
    return new_poly


def reading_img(raster_path):
    with rasterio.open(raster_path, "r", driver="JP2OpenJPEG") as src:
        raster_img = src.read()
        raster_meta = src.meta

    return reshape_as_image(raster_img), raster_meta


def creating_mask(mask_path, raster_path, raster_meta):
    train_df = gpd.read_file(mask_path)

    # let's remove rows without geometry
    train_df = train_df[train_df.geometry.notnull()]

    # assigning crs
    train_df.crs = {"init": "epsg:4284"}

    # transforming polygons to the raster crs
    train_df = train_df.to_crs({"init": raster_meta["crs"]["init"]})

    # rasterize works with polygons that are in image coordinate system
    src = rasterio.open(raster_path, "r", driver="JP2OpenJPEG")

    # creating binary mask for field/not_filed segmentation.
    poly_shp = []
    im_size = (src.meta["height"], src.meta["width"])
    for num, row in train_df.iterrows():
        if row["geometry"].geom_type == "Polygon":
            poly = poly_from_utm(row["geometry"], src.meta["transform"])
            poly_shp.append(poly)
        else:
            for p in row["geometry"]:
                poly = poly_from_utm(p, src.meta["transform"])
                poly_shp.append(poly)

    return rasterize(shapes=poly_shp, out_shape=im_size)


def split_image(image3, tile_size):
    image_shape = tf.shape(image3)
    tile_rows = tf.reshape(image3, [image_shape[0], -1, tile_size[1], image_shape[2]])
    serial_tiles = tf.transpose(tile_rows, [1, 0, 2, 3])
    return tf.reshape(serial_tiles, [-1, tile_size[1], tile_size[0], image_shape[2]])


def unsplit_image(tiles4, image_shape):
    tile_width = tf.shape(tiles4)[1]
    serialized_tiles = tf.reshape(
        tiles4, [-1, image_shape[0], tile_width, image_shape[2]]
    )
    rowwise_tiles = tf.transpose(serialized_tiles, [1, 0, 2, 3])
    return tf.reshape(rowwise_tiles, [image_shape[0], image_shape[1], image_shape[2]])


def pad_image_to_tile_multiple(image3, tile_size, padding="CONSTANT"):
    imagesize = tf.shape(image3)[0:2]
    padding_ = (
        tf.cast(tf.math.ceil(imagesize / tile_size), tf.int32) * tile_size - imagesize
    )
    return tf.pad(image3, [[0, padding_[0]], [0, padding_[1]], [0, 0]], padding)


def create_model(input_size, output_size):
    model = models.Sequential()
    model.add(
        layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=input_size,
            padding="same",
        )
    )
    model.add(
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")
    )
    model.add(
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")
    )
    model.add(
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")
    )
    model.add(layers.UpSampling2D(size=(2, 2)))
    model.add(
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")
    )
    model.add(
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")
    )
    model.add(layers.UpSampling2D(size=(2, 2)))
    model.add(
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")
    )
    model.add(
        layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")
    )
    model.add(
        layers.Conv2D(
            filters=output_size,
            kernel_size=(3, 3),
            activation="sigmoid",
            padding="same",
        )
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(),
        ],
    )

    return model


def main():
    raster_path = "../T36UXV_20200406T083559_TCI_10m.jp2"
    raster_img, raster_meta = reading_img(raster_path)

    mask_path = "../masks/Masks_T36UXV_20190427.shp"
    mask = creating_mask(mask_path, raster_path, raster_meta)

    full_mask = tf.expand_dims(mask, axis=2)  # add color dimention

    tile_size = (180, 180)
    pad_image = pad_image_to_tile_multiple(raster_img, tile_size, padding="CONSTANT")
    pad_mask = pad_image_to_tile_multiple(full_mask, tile_size, padding="CONSTANT")

    image_tiles = split_image(pad_image, tile_size)
    mask_tiles = split_image(pad_mask, tile_size)

    scaled_image = tf.keras.layers.Rescaling(scale=1 / 255)(image_tiles)
    train_image, val_image = scaled_image[:3000], scaled_image[3000:]
    train_mask, val_mask = mask_tiles[:3000], mask_tiles[3000:]

    model = create_model(tf.shape(image_tiles)[1:], tf.shape(mask_tiles)[-1])
    history = model.fit(
        train_image, train_mask, epochs=5, validation_data=(val_image, val_mask)
    )

    predict_mask = model.predict(scaled_image)

    full_predict_mask = unsplit_image(predict_mask, tf.shape(pad_mask))
    full_predict_mask = predict_mask[0 : tf.shape(mask)[0], 0 : tf.shape(mask)[1], :]

    bin_mask_meta = raster_meta.copy()
    bin_mask_meta.update({"count": 1})
    with rasterio.open("../prediction.jp2", "w", **bin_mask_meta) as dst:
        dst.write(full_predict_mask[0, ...] * 255, 1)


if __name__ == "__main__":
    main()
