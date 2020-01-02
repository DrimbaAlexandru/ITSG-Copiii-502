import numpy as np
import nibabel as nib


def load_nifti_image(img_path):
    proxy_img = nib.load( img_path )
    canonical_img = nib.as_closest_canonical(proxy_img)

    image_data = canonical_img.get_fdata()
    return image_data, canonical_img.affine


def save_nifti_image(img_data, affine, img_path):
    img = nib.Nifti1Image(img_data, affine)
    img.to_filename(img_path)
    nib.save(img, img_path)

def load_and_prepare_nifti_image(path):
    image_data, _ = load_nifti_image(path)
    image_data = image_data * ( 255.0 / image_data.max() )
    image_data = image_data.astype(np.uint8)
    image_data = np.expand_dims( image_data, axis=-1 )
    return image_data


def evaluate_model(model, batch_size):
    input_learn = model.train_images[int(model.train_images.shape[0] * model.VALIDATION_SPLIT ):]
    input_val   = model.train_images[:int(model.train_images.shape[0] * model.VALIDATION_SPLIT )]

    masks_learn = model.train_masks[int(model.train_images.shape[0] * model.VALIDATION_SPLIT ):]
    masks_val  = model.train_masks[:int(model.train_images.shape[0] * model.VALIDATION_SPLIT )]

    metrics_learn = model.model.evaluate(input_learn,masks_learn,batch_size=batch_size)
    metrics_val = model.model.evaluate(input_val,masks_val,batch_size=batch_size)

    metrics = {}
    metrics[model.epochs_measured]=[]
    metrics[model.epochs_measured].append(metrics_learn[1:3])
    metrics[model.epochs_measured].append(metrics_val[1:3])

    if model.IS_TEST_DATA_LABELED:
        metrics_test = model.model.evaluate(model.test_images,model.test_masks,batch_size=batch_size)
        metrics[model.epochs_measured].append(metrics_test[1:3])
        
        
def write_model_metrics(model, metrics):
    resultsFile = open(model.LOG_DIR + "\\results.txt", "w")

    resultsFile.write("Number of learning samples: " + str( model.train_images.shape[0] * ( 1 - model.VALIDATION_SPLIT ) ) )
    resultsFile.write("\nNumber of validation samples: " + str( model.train_images.shape[0] * model.VALIDATION_SPLIT ) )
    if model.IS_TEST_DATA_LABELED:
        resultsFile.write("\nNumber of testing samples: " + str( len(model.test_images) ) )

    resultsFile.write("\nLearning results:" )
    resultsFile.write("\n  IoU,   Dice\n")
    for epoch in range(0,model.epochs_measured+1):
        if epoch in metrics:
            for metric in metrics[epoch][0]:
                resultsFile.write( str( metric )+ ", " )
            resultsFile.write("\n")
    resultsFile.write("\n")

    resultsFile.write("\nValidation results:" )
    resultsFile.write("\n  IoU,   Dice\n")
    for epoch in range(0,model.epochs_measured+1):
        if epoch in metrics:
            for metric in metrics[epoch][1]:
                resultsFile.write( str( metric )+ ", " )
            resultsFile.write("\n")
    resultsFile.write("\n")

    if( model.IS_TEST_DATA_LABELED ):
        resultsFile.write("\nTesting results:" )
        resultsFile.write("\n  IoU,   Dice\n")
        for epoch in range(0,model.epochs_measured+1):
            if epoch in metrics:
                for metric in metrics[epoch][2]:
                    resultsFile.write( str( metric )+ ", " )
                resultsFile.write("\n")
        resultsFile.write("\n")

        resultsFile.close()
