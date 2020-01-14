import os
import tkinter as tk

import nibabel as nib
import numpy as np


def load_nifti_image(img_path):
    proxy_img = nib.load(img_path)
    canonical_img = nib.as_closest_canonical(proxy_img)

    image_data = canonical_img.get_fdata()
    return image_data, canonical_img.affine


def save_nifti_image(img_data, affine, img_path):
    img = nib.Nifti1Image(img_data, affine)
    img.to_filename(img_path)
    nib.save(img, img_path)


def load_and_prepare_nifti_image(path):
    image_data, _ = load_nifti_image(path)
    image_data = image_data * (255.0 / image_data.max())
    image_data = image_data.astype(np.uint8)
    image_data = np.expand_dims(image_data, axis=-1)
    return image_data


def get_path_for_saving(extension):
    f = tk.filedialog.asksaveasfile(mode='w', defaultextension=extension)
    if f is None:
        return None
    else:
        return f.name


def evaluate_model(model, metrics, batch_size):
    input_learn = model.train_images[int(model.train_images.shape[0] * model.VALIDATION_SPLIT):]
    input_val = model.train_images[:int(model.train_images.shape[0] * model.VALIDATION_SPLIT)]

    masks_learn = model.train_masks[int(model.train_images.shape[0] * model.VALIDATION_SPLIT):]
    masks_val = model.train_masks[:int(model.train_images.shape[0] * model.VALIDATION_SPLIT)]

    metrics_learn = model.model.evaluate(input_learn, masks_learn, batch_size=batch_size)
    metrics_val = model.model.evaluate(input_val, masks_val, batch_size=batch_size)

    metrics["learn_size"] = len(input_learn)
    metrics["validation_size"] = len(input_val)

    metrics[model.epochs_measured] = []
    metrics[model.epochs_measured].append(metrics_learn[1:3])
    metrics[model.epochs_measured].append(metrics_val[1:3])

    if model.IS_TEST_DATA_LABELED:
        metrics_test = model.model.evaluate(model.test_images, model.test_masks, batch_size=batch_size)
        metrics["test_size"] = len(model.test_images)

        metrics[model.epochs_measured].append(metrics_test[1:3])

    return metrics


def evaluate_model_generator(model, metrics):
    metrics_learn = model.model.evaluate_generator(generator=model.learning_generator)
    metrics_val = model.model.evaluate_generator(generator=model.validation_generator)
    metrics["learn_size"] = model.learning_generator.get_item_count()
    metrics["validation_size"] = model.validation_generator.get_item_count()

    metrics[model.epochs_measured] = []
    metrics[model.epochs_measured].append(metrics_learn[1:3])
    metrics[model.epochs_measured].append(metrics_val[1:3])

    if model.IS_TEST_DATA_LABELED:
        metrics_test = model.model.evaluate_generator(generator=model.test_generator)
        metrics["test_size"] = model.test_generator.get_item_count()

        metrics[model.epochs_measured].append(metrics_test[1:3])


def write_model_metrics(model, metrics):
    os.makedirs(model.LOG_DIR)
    results_file = open(model.LOG_DIR + "\\results.txt", "w")

    results_file.write("Number of learning samples: " + str(metrics["learn_size"]))
    results_file.write("\nNumber of validation samples: " + str(metrics["validation_size"]))
    if model.IS_TEST_DATA_LABELED:
        results_file.write("\nNumber of testing samples: " + str(metrics["test_size"]))

    results_file.write("\nLearning results:")
    results_file.write("\n  IoU,   Dice\n")
    for epoch in range(0, model.epochs_measured + 1):
        if epoch in metrics:
            for metric in metrics[epoch][0]:
                results_file.write(str(metric) + ", ")
            results_file.write("\n")
    results_file.write("\n")

    results_file.write("\nValidation results:")
    results_file.write("\n  IoU,   Dice\n")
    for epoch in range(0, model.epochs_measured + 1):
        if epoch in metrics:
            for metric in metrics[epoch][1]:
                results_file.write(str(metric) + ", ")
            results_file.write("\n")
    results_file.write("\n")

    if model.IS_TEST_DATA_LABELED:
        results_file.write("\nTesting results:")
        results_file.write("\n  IoU,   Dice\n")
        for epoch in range(0, model.epochs_measured + 1):
            if epoch in metrics:
                for metric in metrics[epoch][2]:
                    results_file.write(str(metric) + ", ")
                results_file.write("\n")
        results_file.write("\n")

        results_file.close()
