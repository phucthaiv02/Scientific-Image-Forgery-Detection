# Scientific-Image-Forgery-Detection
Develop methods that can accurately detect and segment copy-move forgeries within biomedical research images.


## Overview
Scientific images are central to published research, but not all of them are honest. Help protect science from fraudulent image manipulation by building models that can detect and segment copy-move forgeries in biomedical images.

## Dataset
The dataset consists of biomedical images that have been manipulated using copy-move forgery techniques. Each image is paired with a ground truth mask that highlights the forged regions if existing otherwise the image is authentic.

Dataset source: [Kaggle - Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection/data)

Dataset structure:
```bash
/data                       # Root directory
    /supplemental_images    # Additional images for training
        image_1.png
        image_2.png
        ...
    /supplemental_masks     # Masks for supplemental images
        image_1.npy
        image_2.npy
        ...
    /train_images           # Primary training images
        /authentic          # Authentic images
            image_1.png
            image_2.png
            ...
        /forged             # Forged images
            image_1.png
            image_2.png
    /train_masks            # Masks for forged images
        image_1.png
        image_2.png
        ...
```