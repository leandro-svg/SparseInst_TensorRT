from detectron2.export.caffe2_modeling import *
from sparseinst import SparseInst
import numpy as np
from matplotlib import pyplot as plt

class Caffe2SparseInst(Caffe2MetaArch):
    def __init__(self, cfg, torch_model):
        assert isinstance(torch_model, SparseInst)
        # torch_model.backbone.size_divisibility = 32
        super().__init__(cfg, torch_model)
        self.torch_model = torch_model
        self.pixel_mean = self.torch_model.pixel_mean/255
        self.pixel_std = self.torch_model.pixel_std/255

    def get_caffe2_inputs(self, batched_inputs):
        inputs = super().get_caffe2_inputs(batched_inputs)
        return inputs[0]/255

    def encode_additional_info(self, predict_net, init_net):
        pass

    def normalizer(self, image):
        image = (image - self.pixel_mean) / self.pixel_std
        return image

    @mock_torch_nn_functional_interpolate()
    def forward(self, inputs):
        images = self.normalizer(inputs)
        images = ImageList.from_tensors([images], 32)[0]
        # forward
        features = self.torch_model.backbone(images)
        features = self.torch_model.encoder(features)
        output = self.torch_model.decoder(features)
        pred_scores = output["pred_logits"].sigmoid()
        pred_masks = output["pred_masks"].sigmoid()
        pred_objectness = output["pred_scores"].sigmoid()
        pred_scores2 = torch.sqrt(pred_scores * pred_objectness)

        # scores, masks = np.squeeze(pred_scores2), np.squeeze(pred_masks)
        # keep = torch.argmax(scores, axis=1)
        # masks = [masks[label, :, :] for i, label in enumerate(keep) if scores[i, label] > 0.35]
        # fig = plt.figure()
        # num_masks = len(masks)
        # for i, mask in enumerate(masks, 1):
        #     fig.add_subplot(1, num_masks, i)
        #     plt.imshow(mask.data.cpu())
        # plt.show()
        # plt.ion()
        
        # return

        return pred_scores2, pred_masks

    @staticmethod
    def get_outputs_converter(predict_net, init_net):
        pass


META_ARCH_CAFFE2_EXPORT_TYPE_MAP['SparseInst'] = Caffe2SparseInst