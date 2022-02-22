import numpy as np
import torch
import cv2
from climatehack import BaseEvaluator
# from model import Encoder

N_IMS = 24
def average_flows(input_data, **args):
    flows = []
    for i in range(args['n_warmup']):
        temp = dict(args)
        del temp['n_warmup']
        flows.append(cv2.calcOpticalFlowFarneback(prev=input_data[i], next=input_data[i+1], flow=None, **temp))
    flows = np.stack(flows).astype(np.float32)
    return np.average(flows, axis=0, weights=range(1, args['n_warmup']+1)).astype(np.float32)

def flow(im1, im2, **args):
    return cv2.calcOpticalFlowFarneback(prev=im1, next=im2, **args)

def remap_image(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    # Adapted from https://github.com/opencv/opencv/issues/11068
    height, width = flow.shape[:2]
    remap = -flow.copy()
    remap[..., 0] += np.arange(width)  # x map
    remap[..., 1] += np.arange(height)[:, np.newaxis]  # y map
    return cv2.remap(src=image, map1=remap, map2=None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def eval_flow(input_data, args):
    flow1 = average_flows(input_data, **args)
    predicted_l = torch.randn((1, 0, 64, 64))
    last_image = input_data[-1]
    for _ in range(24):
        # flow1 = flow(input_data[i], input_data[i+1], flow=last_flow, **args)
        predicted = remap_image(last_image, flow1)
        i2 = torch.from_numpy(predicted[32:96, 32:96]).view(1, 1, 64, 64)
        last_image = predicted
        predicted_l = torch.cat([predicted_l, i2], dim=1)
    return predicted_l

class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.

        In this case, it loads the trained model (in evaluation mode)."""

        # self.model = Encoder(inputs=12, outputs=N_IMS)
        # self.model.load_state_dict(torch.load('encoder3.pt', map_location=torch.device('cpu')))
        # self.model.eval()
        # self.model = torch.load('encoder2.pt')
        pass

    def predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Makes a prediction for the next two hours of satellite imagery.

        Args:
            coordinates (np.ndarray): the OSGB x and y coordinates (2, 128, 128)
            data (np.ndarray): an array of 12 128*128 satellite images (12, 128, 128)

        Returns:
            np.ndarray: an array of 24 64*64 satellite image predictions (24, 64, 64)
        """

        assert coordinates.shape == (2, 128, 128)
        assert data.shape == (12, 128, 128)
        def weighted_average(weights, x):
            return sum([weights[i]*x[i] for i in range(len(x))])/sum(weights)
        with torch.no_grad():
            # prediction = (
            #     self.model(torch.from_numpy(data).view(-1, 12 * 128 * 128))
            #     .view(24, 64, 64)
            #     .detach()
            #     .numpy()
            # )
            # weights = torch.tensor([-0.14, 1.14])
            # temp = torch.from_numpy(data[-2:, ::, ::])
            # starter = data[-3:, 32:96, 32:96]
            # weights = [0.3115, 0.1378, 0.5507]
            # for i in range(24):
            #     next_im = weighted_average(weights, starter[-3:, ::, ::]).reshape((1, 64, 64))
            #     # print(next_im.shape)
            #     starter = np.append(starter, next_im, axis=0)
            #     # print(starter.shape)
            # prediction = starter[-24:, ::, ::]
            # prediction = torch.tensor(weighted_average([0.1, -0.1, 1], data[-3:, 32:96, 32:96])).view(1, 64, 64).repeat(24, 1, 1).detach().numpy()
            # prediction = (torch.tensor(data[-1][32:96, 32:96]*1.14 + data[-2][32:96, 32:96]*-0.14).view(1, 64, 64).repeat(24, 1, 1)).detach().numpy()
            # best = {'pyr_scale': 0.6, 'levels': 2, 'winsize': 4, 'iterations': 9, 'poly_n': 3, 'poly_sigma': 1.1, 'flags': 256, 'n_warmup': 7}
            # best = {'pyr_scale': 0.5, 'levels': 3, 'winsize': 4, 'iterations': 9, 'poly_n': 2, 'poly_sigma': 0.9, 'flags': 256, 'n_warmup': 7}
            # best = {'pyr_scale': 0.7, 'levels': 3, 'winsize': 4, 'iterations': 9, 'poly_n': 3, 'poly_sigma': 0.6, 'flags': 256, 'n_warmup': 7}
            best = {'pyr_scale': 0.7, 'levels': 6, 'winsize': 4, 'iterations': 24, 'poly_n': 5, 'poly_sigma': 1, 'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN, 'n_warmup': 7}
            # flow = average_flows(data, 6, best)
            prediction = eval_flow(data, best).view(24, 64, 64).numpy()
    #         flow = average_flows(data, 6, pyr_scale=0.7, levels=2, winsize=10, iterations=10, 
    # poly_n=3, poly_sigma=0.7, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    #         starter = data[-1]
    #         prediction = []
    #         for _ in range(24):
    #             starter = remap_image(starter, flow)
    #             starter_draw = starter[32:96,32:96]
    #             prediction.append(starter_draw)
    #         prediction = np.array(prediction)
            # print(prediction.shape)

            #OLD ENCODER STUFF--------
            # im_sequence = torch.tensor(data).view(1, 12, 128, 128)#.to(device)
            # predicted_tensor = self.model(im_sequence).detach()
            # # initial_prediction = predicted_tensor[::, ::, 32:96, 32:96]#.view(12, 64, 64)#.numpy()
            # initial_prediction = predicted_tensor[::, :12, 32:96, 32:96]#.view(12, 64, 64)#.numpy()
            # # # initial_prediction = predicted_tensor[::, ::, 32:96, 32:96].view(24, 64, 64).numpy()
            # repeated = torch.mean(initial_prediction[::, :3, ::, ::], dim=1, keepdim=True).repeat(1, 24 - initial_prediction.shape[1], 1, 1)
            # # # repeated = torch.mean(initial_prediction[::, :3, ::, ::], dim=1, keepdim=True).repeat(1, 12, 1, 1)
            # prediction = torch.cat([initial_prediction, repeated], dim=1).view(24, 64, 64).numpy()
            #---------------

            # prediction = initial_prediction.view(24, 64, 64).numpy()
            
            # starter = im_sequence
            # for i in range(23//N_IMS + 1):
            #     predicted_tensor = self.model(starter[::, -N_IMS:, ::, ::].view(1, N_IMS, 128, 128)).detach()
            #     starter = torch.cat([starter, predicted_tensor], dim=1)
            # prediction = starter[::, N_IMS:, 32:96, 32:96].view(24, 64, 64).numpy()
            assert prediction.shape == (24, 64, 64)
            return prediction


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
