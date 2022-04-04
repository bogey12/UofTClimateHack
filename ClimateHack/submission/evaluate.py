import numpy as np
import torch
import torchvision
from climatehack import BaseEvaluator
from model import ConvLSTMModel, UNET, UNETViT
from pytorch_msssim import ms_ssim, MS_SSIM


class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.

        In this case, it loads the trained model (in evaluation mode)."""

        self.mean = 0.3028213
        self.stddev = 0.16613534
        #self.model = UNET(12,24)
        self.model = UNETViT(12,24)
        self.model.load_state_dict(torch.load("Test_UNet_ViT_V2.pth"))
        self.model.eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    # Seq to seq UNET prediction
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

    # Seq to one UNET prediction
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


    def predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        # Commented Code for persistence baseline
        
        #batch_predictions = []
        #for i in range(0,24):
        #    batch_predictions.append(data[-1,32:96,32:96])
        #return np.stack(batch_predictions,axis=0)
        return self.UNET_predict(coordinates,data)


def main():
    evaluator = Evaluator()
    evaluator.evaluate()

if __name__ == "__main__":
    main()
