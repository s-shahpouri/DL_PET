import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm


class Preprocessor:
    def __init__(self,json_parameters,data_path,save_path):
        self.save_path = save_path
        self.data_path = data_path
        self.json_path = json_parameters
        
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    def read_image(self, img_fn, orientation, background_value):
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_fn)).astype(np.float32)
        img = self.rotate_image(img, orientation)
        img = self.reshape_image(img, background_value)
        return img
    
    def rotate_image(self, img, orientation):
        """
        This function swaps axes in order to have the data in SAC format.
        """

        if orientation == "axial": # Sagittal, Axial, Coronal
            return img
        elif orientation == "sagittal": # Axial, Coronal, Sagittal:
            m1 = np.swapaxes(img, 0, 2) # CAS
            m1 = np.rot90(m1, k=1, axes=(1,2))
            return m1
        elif orientation == "coronal":
            m1 = np.swapaxes(img, 0, 1) # CAS
            m1 = np.rot90(m1, k=2, axes=(1,2))
            return m1
        else:
            print("Unknow orientation...Exit")
            exit()

    def reshape_image(self, img, background_value=0.0):
        """
        This function reshapes the image to a cubic matrix.
        """

        img_shape = img.shape
        max_image_shape = 512
        offsets = (int(np.floor(max_image_shape-img_shape[0])/2.0), int(np.floor(max_image_shape-img_shape[1])/2.0), int(np.floor(max_image_shape-img_shape[2])/2.0))

        reshaped_img = np.ones((img_shape[0], max_image_shape, max_image_shape), dtype=np.float32) * float(background_value)
        reshaped_img[:,offsets[1]:offsets[1]+img_shape[1],offsets[2]:offsets[2]+img_shape[2]]=img[:,:,:]
        
        # reshaped_img = np.ones((max_image_shape, max_image_shape, max_image_shape), dtype=np.float32) * float(background_value)
        # reshaped_img[offsets[0]:offsets[0]+img_shape[0],offsets[1]:offsets[1]+img_shape[1],offsets[2]:offsets[2]+img_shape[2]]=img[:,:,:]

        return reshaped_img.astype(np.float32)

    def preprocess(self,case_list):
        progressbar = tqdm(case_list)
        for i,case in enumerate(progressbar):
            progressbar.set_description(f'Preprocessing image: {case}')
            train_case_path = os.path.join(self.data_path,case)
            Y= self.read_image(os.path.join(train_case_path, self.json_path["Y"]), self.json_path["orientation"], background_value=-1000) # This is for CT
            VOI = self.read_image(os.path.join(train_case_path, self.json_path["VOI"]), self.json_path["orientation"], background_value=0) # Volume of Interst based on skin mask
            # VOI = VOI >0
            X = self.read_image(os.path.join(train_case_path, self.json_path['X']), self.json_path["orientation"], background_value=-1000) # what is the additional channel index for

            for slice in range(X.shape[0]):
                # if VOI[slice].max() >0:
                    slice_X = X[slice]
                    slice_Y = Y[slice]
                    slice_voi = VOI[slice]
                    if slice < 10:
                        np.save(os.path.join(self.save_path,f'{case}_00{slice}_X'),slice_X)
                        np.save(os.path.join(self.save_path,f'{case}_00{slice}_Y'),slice_Y)
                        np.save(os.path.join(self.save_path,f'{case}_00{slice}_mask'),slice_voi)
                    elif slice < 100:
                        np.save(os.path.join(self.save_path,f'{case}_0{slice}_X'),slice_X)
                        np.save(os.path.join(self.save_path,f'{case}_0{slice}_Y'),slice_Y)
                        np.save(os.path.join(self.save_path,f'{case}_0{slice}_mask'),slice_voi)
                    else:
                        np.save(os.path.join(self.save_path,f'{case}_{slice}_X'),slice_X)
                        np.save(os.path.join(self.save_path,f'{case}_{slice}_Y'),slice_Y)
                        np.save(os.path.join(self.save_path,f'{case}_{slice}_mask'),slice_voi)