"""
    A test script to get WevadeBQ on.
"""
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import math, torch, cv2, argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from art.estimators.classification import PyTorchClassifier

# === Project Import ===
from utils.general import uint8_to_float, img_np_to_tensor, float_to_uint8, save_image_bgr,\
    tensor_output_to_image_np, watermark_np_to_str
from utils.hop_skip_jump import HopSkipJump
from watermarkers import get_watermarkers
from evations import get_evasion_alg
from utils.plottings import plot_dip_res, plot_vae_res, plot_corruption_res, \
    plot_diffuser_res
from noise_layers.diff_jpeg import DiffJPEG


# Rewrite The decider into tensor wise operation
class WMTensorDetector(nn.Module):
    
    def __init__(self, gt, watermarker, th=0.8) -> None:
        super().__init__()
        self.th = th
        self.decoder = watermarker
        self.gt = torch.from_numpy(gt[np.newaxis, :])
        # self.unused_layer_to_pass_check = nn.Linear(in_features=1, out_features=1)

    def forward(self, input_tensor, verbose=False):
        # === Input tensor needs to be [0, 1] tensor
        input_tensor = torch.clamp(input_tensor, 0, 1)
        input_array = (input_tensor.detach().cpu().numpy() * 255).astype(np.uint8)[0, :, :, :]
        input_array = np.transpose(input_array, [1, 2, 0])
        decoded_message = self.decoder.decode(input_array)[np.newaxis, :]
        print_msg = watermark_np_to_str(decoded_message[0, :])
        decoded_message = torch.from_numpy(decoded_message)
        bit_acc = 1 - torch.sum(torch.abs(decoded_message-self.gt), 1)/self.gt.shape[1]
        class_idx = torch.logical_or((bit_acc>self.th), (bit_acc<(1-self.th))).long()
        if verbose:
            print("Decoded smg: %s" % print_msg)
            print("Bitwise acc: {:.02f}".format(bit_acc.item() * 100))
        return F.one_hot(class_idx, num_classes=2)

    def predict(self, input_array, device=torch.device("cpu")):
        input_tensor = torch.from_numpy(input_array).to(device)
        with torch.no_grad():
            return self.forward(input_tensor).cpu().numpy()


# Note: watermarked_images, labels and adv_images are in numpy arrays
def JPEG_initailization(watermarked_images_tensor, labels, detector, quality_ls, verbose=True, device=torch.device("cpu")):
    # JPEG initialization
    adv_images = watermarked_images_tensor.to(device)
    detector = detector

    flag = False            # whether an adversarial example has been found
    for quality in quality_ls: 
        print("Init JPEG Quality: {}".format(quality))
        jpeg_module = DiffJPEG(quality=quality).to(device)

        jpeg_image = adv_images.clone()
        jpeg_image_max = torch.max(jpeg_image)
        jpeg_image_min = torch.min(jpeg_image)
        jpeg_image = (jpeg_image-jpeg_image_min)/(jpeg_image_max-jpeg_image_min)
        jpeg_image = jpeg_module(jpeg_image.to(device))
        jpeg_image = jpeg_image*(jpeg_image_max-jpeg_image_min)+jpeg_image_min
        pred = detector.forward(jpeg_image, verbose=True).detach().cpu().numpy()
        pred = np.argmax(pred,-1)[0]
        if pred!=labels: # succeed
            adv_images = jpeg_image
            flag = True
            break
        del jpeg_module
    print("Finish JPEG Initialization.")
    
    if verbose:
        print("Init JPEG corruption evade the watermark: {}".format(flag))

    return adv_images



# Note: watermarked_images, init_adv_images and best_adv are in numpy arrays
# def WEvade_B_Q(args, watermarked_images, init_adv_images, detector, num_queries_ls, verbose=True):
def WEvade_B_Q(watermarked_images, init_adv_images, detector, verbose=True):
    # num_images = len(watermarked_images) 
    num_images = 1  # For the sake of code unity, fix this to be 1


    norm = 2
    attack = HopSkipJump(
        classifier=detector, targeted=False, norm=norm, max_iter=0, max_eval=1000, init_eval=5, batch_size=1
    )
    
    total_num_queries = 0
    saved_num_queries_ls = [0]
    es_ls = np.zeros((num_images))    # a list of 'es' in Algorithm 3
    es_flags = np.zeros((num_images)) # whether the attack has been early stopped
    num_natural_adv = 0
    num_early_stop = 0
    num_regular_stop = 0

    adv_images = init_adv_images.copy()
    best_adv = init_adv_images
    best_norms = np.ones((num_images))*1e8

    ### Algorithm
    max_iterations = 1000 # a random large number
    for i in range(int(max_iterations/1)):

        adv_images, num_queries_ls = attack.generate(
            x=watermarked_images, x_adv_init=adv_images, num_queries_ls=saved_num_queries_ls, resume=True
        ) # use resume to continue previous attack

        if verbose:
            print("Step: {}; Number of queries: {}".format((i * 1), num_queries_ls))

        # save the best results
        avg_error = 0
        for k in range(len(adv_images)):
            if norm == 'inf':
                error = np.max(np.abs(adv_images[k] - watermarked_images[k]))
            else:
                error = np.linalg.norm(adv_images[k] - watermarked_images[k])

            if es_flags[k]==0: # update if the attack has not been early stopped
                if error<best_norms[k]:
                    best_norms[k] = error
                    best_adv[k] = adv_images[k]
                    es_ls[k] = 0
                else:
                    es_ls[k]+=1
            avg_error += best_norms[k]
        avg_error = avg_error/2 # [-1,1]->[0,1]
        if verbose:
            print("Adversarial images at step {}.".format(i * 1))
            print("Average best error in l_{} norm: {}\n".format(norm, avg_error/len(adv_images)))

        # stopping criteria
        # natural_adv
        for k in range(len(adv_images)):
            if best_norms[k]==0 and es_flags[k]==0:
                es_flags[k] = 1
                total_num_queries += 0
                saved_num_queries_ls[k] = 0
                num_natural_adv += 1
        # regular_stop
        for k in range(len(adv_images)):
            if num_queries_ls[k]>=2000 and es_flags[k]==0:
                es_flags[k] = 1
                total_num_queries += num_queries_ls[k]
                saved_num_queries_ls[k] = num_queries_ls[k]
                num_regular_stop+=1
        # early_stop
        for k in range(len(adv_images)):
            if es_ls[k]==20 and es_flags[k]==0:
                es_flags[k] = 1
                total_num_queries += num_queries_ls[k]
                saved_num_queries_ls[k] = num_queries_ls[k]
                num_early_stop += 1

        if np.sum(es_flags==0)==0:
            break
        attack.max_iter = 1

    assert np.sum(es_flags)==num_images
    assert num_natural_adv+num_regular_stop+num_early_stop==num_images
    assert np.sum(saved_num_queries_ls)==total_num_queries
    del attack

    if verbose:
        print("Number of queries used for each sample:")
        print(saved_num_queries_ls)

    return best_adv, saved_num_queries_ls


def wevade_bq(
    img_clean_path, img_w_path,  watermarker, watermark_gt, evader_cfgs, vis_root
):
    device = evader_cfgs["device"]
    dtype = evader_cfgs["dtype"]

    img_w_bgr = cv2.imread(img_w_path)
    img_w_bgr_float = uint8_to_float(img_w_bgr)
    img_w_bgr_tensor = img_np_to_tensor(img_w_bgr_float).to(device, dtype=dtype)
    
    detector = WMTensorDetector(watermark_gt, watermarker)
    quality_ls = [99,90,70,50,30,10,5,3,2,1]

    init_adv_images_tensor = JPEG_initailization(
        img_w_bgr_tensor, np.asarray([1]), detector, quality_ls,verbose=True, device=device
    )

    # Vis InitAdv Image
    vis_img = float_to_uint8(tensor_output_to_image_np(init_adv_images_tensor))
    save_name = os.path.join(vis_root, "img_init_adv.png")
    save_image_bgr(vis_img, save_name)

    # === WeVadeBQ ===
    img_w_bq_input = img_w_bgr_tensor.detach().cpu().numpy()
    im_size = 512
    bq_detector =  PyTorchClassifier(
        model=detector,
        clip_values=(0, 1.0),
        input_shape=(3, im_size, im_size),
        nb_classes=2,
        use_amp=False,
        channels_first=True,
        loss=None,
    )
    best_adv_images, saved_num_queries_ls = WEvade_B_Q(
        img_w_bq_input, init_adv_images_tensor.detach().cpu().numpy(), bq_detector, verbose=evader_cfgs["verbose"])
    print(np.amin(best_adv_images), np.amax(best_adv_images))

    im_res = np.transpose(best_adv_images[0, :, :, :], [1, 2, 0])
    im_res_bgr = float_to_uint8(im_res)
    save_name = os.path.join(vis_root, "hhjumped_final.png")
    save_image_bgr(im_res_bgr, save_name)

    # === Sanity check ===
    final_decode = watermarker.decode(im_res_bgr)
    bitwise_acc = np.mean(watermark_gt == final_decode)
    print("Hip-Hop-Jumped decoded watermark: ")
    final_decode_str = watermark_np_to_str(final_decode)
    print("  {}".format(final_decode_str))
    print("  Bitwise acc: [%.02f]".format(bitwise_acc * 100))


def main(args):
    # === Some Dummy Configs ===
    device = torch.device("cuda")
    # === Get image paths                          
    img_clean_path = os.path.join(
        args.root_path_im_orig, args.im_name  # Path to a clean image
    )

    img_w_root_dir = os.path.join(
        args.root_path_im_w, args. watermarker
    )
    os.makedirs(img_w_root_dir, exist_ok=True)
    img_w_path = os.path.join(
        img_w_root_dir, args.im_name  # Path to save the watermarked image.
    )  

    # === Initiate a watermark ==> in ndarray
    watermark_gt = np.random.binomial(1, 0.5, 32)
    # === Initiate a encoder & decoder ===
    watermarker_configs = {
        "watermarker": args.watermarker,
        "watermark_gt": watermark_gt
    }
    watermarker = get_watermarkers(watermarker_configs)

    # Generated watermarked image and save it to img_w_path
    watermarker.encode(img_clean_path, img_w_path)
    # Check decoding in case learning-based encoder/decoder doesn't work properly
    watermark_decode = watermarker.decode_from_path(img_w_path)
    bitwise_acc_0 = np.mean(watermark_decode == watermark_gt)
    print("** Sanity check for watermarker encoder & decoder:")
    print("    Bitwise acc. - [{:.04f} %]".format(bitwise_acc_0 * 100))
    assert bitwise_acc_0 > 0.99, "The encoder & decode fails to work on this watermark string."
    # ##### #### #### #### ######

    # === Get Evasion algorithm ===
    detection_threshold = args.detection_threshold
    print("Setting detection threshold [{:02f}] for the watermark detector.".format(detection_threshold))
    # Create log folder 
    evader_cfgs = {
        "arch": "dummy",
        "device": device,
        "dtype": torch.float,
        "detection_threshold": detection_threshold,
        "verbose": True,    
    }
    vis_root_dir = os.path.join(
        ".", "Visualization", "{}".format(args.im_name.split(".")[0]), "{}".format(args.watermarker), "{}".format(args.evade_method), "{}".format(evader_cfgs["arch"])
    )
    os.makedirs(vis_root_dir, exist_ok=True)
    
    evasion_res = wevade_bq(
        img_clean_path, img_w_path,  watermarker, watermark_gt, evader_cfgs, vis_root_dir
    )



if __name__ == "__main__": 

    print("\n***** This is demo of single image evasion ***** \n")
    
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        '--root_path_im_orig', type=str, help="Root folder to the clean images.",
        default=os.path.join("examples", "ori_imgs")
    )
    parser.add_argument(
        "--im_name", dest="im_name", type=str, help="clean image name.",
        default="711.png"
    )
    parser.add_argument(
        "--root_path_im_w", dest="root_path_im_w", type=str, help="Root folder to save watermarked image.",
        default=os.path.join("examples", "watermarked_imgs")
    )

    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, help="Specification of watermarking method.",
        default="rivaGan"
    )
    parser.add_argument(
        "--evade_method", dest="evade_method", type=str, help="Specification of evasion method.",
        default="Wevadebq"
    )
    parser.add_argument(
        "--arch", dest="arch", type=str, help="Secondary specification of evasion method (if there are other choices).",
        default="dummy"
    )
    parser.add_argument(
        "--detection_threshold", dest="detection_threshold", type=float, default=0.75,
        help="Tunable threhsold to check if the evasion is successful."
    )
    args = parser.parse_args()
    main(args)

    print("\n***** Completed. *****\n")