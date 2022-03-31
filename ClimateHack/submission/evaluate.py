import numpy as np
#import tensorflow as tf
import torch
import torchvision
#import mymetnet
from climatehack import BaseEvaluator
from model import ConvLSTMModel, UNET, UNETViT
from pytorch_msssim import ms_ssim, MS_SSIM


class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.

        In this case, it loads the trained model (in evaluation mode)."""

        
        #self.model = tf.keras.models.load_model('saved_model/my_model')
        self.mean = 0.3028213
        self.stddev = 0.16613534
        #self.model = UNET(12,24)
        self.model = UNETViT(12,24)
        self.model.load_state_dict(torch.load("Test_UNet_ViT_V2.pth"))
        #self.model.load_state_dict(torch.load("Unet_ViTBilinear_Crossattend.pth"))
        #self.model.eval()
        #self.model = torch.jit.load("../UNet_ViT_CBAM_proj12_ResizeConv.pt")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def ConvLSTMpredict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Makes a prediction for the next two hours of satellite imagery.

        Args:
            coordinates (np.ndarray): the OSGB x and y coordinates (2, 128, 128)
            data (np.ndarray): an array of 12 128*128 satellite images (12, 128, 128)

        Returns:
            np.ndarray: an array of 24 64*64 satellite image predictions (24, 64, 64)
        """

        assert coordinates.shape == (2, 128, 128)
        assert data.shape == (12, 128, 128)
        
        batch_features = data / 1023
        batch_features = (data - self.mean) / self.stddev
        with torch.no_grad():
            batch_features = torch.Tensor(batch_features).unsqueeze(0)
            batch_features = batch_features.to(device)
            # crop_data (1,1,1,64,64)
            crop_data = batch_features[:,:,-1,32:96,32:96].unsqueeze(0)
            cur_input = crop_data
            prediction = []
            for i in range(24):
                # out = [1,1,1,64,64]
                out = self.model(cur_input)
                cur_input = out
                # Revert output to original image
                out_img = out.squeeze() # (64,64)
                out_img = out_img * self.stddev + self.mean
                out_img = out_img * 1023
                prediction.append(out_img.numpy())

        prediction = np.array(prediction)
        assert prediction.shape == (24, 64, 64)
        return prediction

    def UNET_predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        assert coordinates.shape == (2, 128, 128)
        assert data.shape == (12, 128, 128)
        
        batch_features = data / 1023
        batch_features = (batch_features - self.mean) / self.stddev
       
        with torch.no_grad():
            batch_features = torch.Tensor(batch_features).unsqueeze(0)
            batch_features = batch_features.to(self.device)
            batch_predictions = self.model(batch_features)
            #batch_predictions = batch_predictions.view(-1,1,24,128,128).squeeze(1)
            batch_predictions = batch_predictions[:,:,32:96,32:96]
            batch_predictions = batch_predictions.squeeze()
            prediction = 1023 * batch_predictions.cpu().numpy()


        assert prediction.shape == (24, 64, 64)
        return prediction

    def UNET_seq2one(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        assert coordinates.shape == (2, 128, 128)
        assert data.shape == (12, 128, 128)
        
        batch_features = data / 1023
        batch_features = (batch_features - self.mean) / self.stddev
       
        with torch.no_grad():
            batch_features = torch.Tensor(batch_features).unsqueeze(0)
            batch_features = batch_features.to(self.device)
            batch_predictions = []
            for i in range(0,24):
                temp_out = self.model(batch_features)
                batch_predictions.append(temp_out)
                new_features = (temp_out - self.mean) / self.stddev
                batch_features = torch.cat((batch_features[:,1:12,:,:], new_features.detach()), axis=1)
            batch_predictions = torch.stack(batch_predictions, dim=2)
            batch_predictions = batch_predictions[:,:,:,32:96,32:96]
            batch_predictions = batch_predictions.squeeze()
            prediction = 1023 * batch_predictions.cpu().numpy()
        assert prediction.shape == (24, 64, 64)
        return prediction
    def metnet_predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Makes a prediction for the next two hours of satellite imagery.

        Args:
            coordinates (np.ndarray): the OSGB x and y coordinates (2, 128, 128)
            data (np.ndarray): an array of 12 128*128 satellite images (12, 128, 128)

        Returns:
            np.ndarray: an array of 24 64*64 satellite image predictions (24, 64, 64)
        """

        assert coordinates.shape == (2, 128, 128)
        assert data.shape == (12, 128, 128)
        
        batch_features = data / 1023
        batch_features = (batch_features - self.mean) / self.stddev
       
        with torch.no_grad():
            batch_features = torch.Tensor(batch_features).unsqueeze(0).unsqueeze(0)
            batch_features = batch_features.to(self.device)
            # shape (1,1,12,64,64)
            # MetNet Train Flow (permute from (1,1,12,128,128) -> (1,12,1,128,128))
            batch_features = batch_features.permute(0, 2, 1, 3, 4)
            batch_features = batch_features[:,:,:,32:96,32:96]
            batch_predictions = []
            for lead_time in range(24):
                tmp_out = self.model(batch_features, 0)
                batch_predictions.append(tmp_out)
                new_features = (tmp_out - self.mean) / self.stddev
                new_features = new_features.squeeze(1).view(-1,1,1,64,64)
                batch_features = torch.cat((batch_features[:,1:12,:,:,:], new_features.detach()), axis=1)
            batch_predictions = torch.stack(batch_predictions, dim=1)
            # predictions shape (1,24,64,8,8) -> (1,24,1,64,64)
            batch_predictions = batch_predictions.view(-1,24,1,64,64)
            # Repermute batch_predictions: (1,24,1,64,64) -> (24,64,64)
            batch_predictions = batch_predictions.squeeze()
            prediction = 1023 * batch_predictions.cpu().numpy()
        assert prediction.shape == (24, 64, 64)
        return prediction

    def metnetseq2one_predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Makes a prediction for the next two hours of satellite imagery.

        Args:
            coordinates (np.ndarray): the OSGB x and y coordinates (2, 128, 128)
            data (np.ndarray): an array of 12 128*128 satellite images (12, 128, 128)

        Returns:
            np.ndarray: an array of 24 64*64 satellite image predictions (24, 64, 64)
        """

        assert coordinates.shape == (2, 128, 128)
        assert data.shape == (12, 128, 128)
        #expand = torchvision.transforms.Resize([128,128])
        batch_features = data / 1023
        batch_features = (batch_features - self.mean) / self.stddev
        #batch_features = data
        with torch.no_grad():
            batch_features = torch.Tensor(batch_features).unsqueeze(0).unsqueeze(0)
            batch_features = batch_features.to(self.device)
            #batch_features = batch_features[:,:,:,32:96,32:96]
            # shape (1,1,12,64,64)
            # MetNet Train Flow (permute from (1,1,12,128,128) -> (1,12,1,128,128))
            batch_features = batch_features.permute(0, 2, 1, 3, 4)
            batch_predictions = []
            cur_in = batch_features
            for i in range(24):
                temp_out = self.model(cur_in, 0)
                # predictions shape (1,1,64,8,8) -> (1,1,1,64,64)
                temp_out = temp_out.view(-1,1,1,64,64)
                batch_predictions.append(temp_out)
                # resize temp out to (1,1,1,128,128)
                #cur_out = expand(temp_out.squeeze(0)).unsqueeze(0)
                # concatenate (1,11,1,128,128) with (1,1,1,128,128) on axis 1
                #new_features = (temp_out - self.mean) / self.stddev
                #cur_in = torch.cat((cur_in[:,1:12,:,:,:], new_features), axis=1)
            batch_predictions = torch.stack(batch_predictions, dim=1)
            batch_predictions = batch_predictions.squeeze()
            prediction = 1023 * batch_predictions.cpu().numpy()
        assert prediction.shape == (24, 64, 64)
        return prediction

    def predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        batch_predictions = []
        for i in range(0,24):
            batch_predictions.append(data[-1,32:96,32:96])
        return np.stack(batch_predictions,axis=0)
        #return self.UNET_predict(coordinates,data)


def main():
    evaluator = Evaluator()
    #evaluator.evaluate()

if __name__ == "__main__":
    main()
