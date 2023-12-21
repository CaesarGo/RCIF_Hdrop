# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode


############################ hook #################cyj

def my_hook(module, input, output):
    print("hook added")  #cyj


lose_571 = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 22, 23, 25, 27, 29, 30, 34, 35, 36, 37, 38, 39, 41, 43, 44, 45, 46, 47, 48, 50, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 65, 66, 67, 68, 69, 71, 72, 74, 75, 76, 77, 79, 80, 82, 84, 88, 93, 96, 97, 99, 104, 105, 106, 109, 110, 111, 112, 114, 115, 116, 120, 122, 126, 130, 131, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 145, 146, 147, 149, 150, 152, 153, 154, 155, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 177, 178, 179, 180, 181, 182, 183, 185, 187, 188, 189, 190, 191, 193, 194, 195, 196, 198, 200, 201, 202, 203, 204, 205, 207, 208, 209, 210, 211, 212, 215, 216, 218, 219, 221, 224, 243, 246, 249, 257, 258, 265, 269, 277, 280, 282, 286, 288, 290, 291, 292, 297, 299, 303, 309, 312, 313, 317, 320, 322, 328, 330, 333, 334, 335, 338, 339, 341, 342, 343, 344, 350, 352, 353, 355, 357, 358, 362, 364, 365, 369, 370, 374, 378, 388, 391, 396, 397, 398, 399, 403, 404, 407, 410, 417, 422, 425, 428, 430, 434, 435, 437, 439, 440, 442, 444, 446, 448, 452, 459, 460, 461, 462, 465, 466, 468, 469, 471, 472, 473, 474, 480, 481, 482, 483, 491, 492, 498, 509, 511, 515, 520, 521, 522, 524, 525, 526, 527, 528, 529, 530, 531, 532, 534, 535, 536, 537, 538, 539, 541, 543, 544, 545, 546, 547, 548, 550, 551, 552, 555, 556, 558, 560, 561, 563, 564, 565, 566, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 580, 581, 582, 583, 584, 585, 586, 587, 588, 590, 591, 592, 594, 595, 596, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 614, 615, 617, 618, 619, 620, 621, 629, 631, 635, 639, 641, 644, 646, 650, 653, 654, 655, 657, 659, 664, 665, 670, 672, 675, 676, 677, 679, 680, 681, 683, 684, 685, 686, 690, 692, 693, 694, 695, 696, 697, 698, 701, 704, 706, 709, 713, 715, 716, 718, 719, 720, 724, 731, 733, 734, 735, 736, 740, 741, 745, 749, 750, 756, 758, 763, 764, 766, 770, 772, 773, 785, 786, 788, 789, 790, 791, 795, 797, 798, 801, 802, 805, 806, 809, 811, 815, 817, 819, 820, 822, 823, 824, 825, 828, 831, 832, 833, 834, 835, 836, 837, 839, 840, 841, 842, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 933, 934, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 947, 948, 952, 953, 954, 955, 958, 959, 960, 961, 962, 963, 964, 965, 967, 968, 970, 971, 972, 976, 978, 979, 980, 981, 982, 983, 984, 985, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998]
#lose_0_1 = torch.arange(1,1000,2)
lose_0 = []
lose_406 =[0, 3, 7, 8, 11, 12, 13, 16, 18, 32, 33, 35, 36, 40, 42, 43, 44, 45, 46, 48, 49, 50, 52, 59, 64, 67, 71, 74, 79, 84, 88, 90, 97, 100, 104, 108, 110, 116, 117, 119, 121, 123, 127, 128, 129, 131, 136, 140, 144, 147, 150, 152, 153, 154, 155, 156, 157, 158, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 172, 173, 174, 175, 176, 177, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 198, 199, 200, 201, 202, 204, 206, 207, 210, 211, 212, 213, 214, 215, 216, 218, 220, 221, 223, 224, 225, 226, 227, 228, 229, 232, 233, 234, 235, 236, 238, 239, 241, 242, 243, 246, 250, 251, 252, 253, 254, 260, 263, 264, 265, 266, 267, 268, 270, 271, 273, 275, 276, 277, 278, 281, 282, 283, 284, 292, 296, 300, 301, 307, 309, 314, 317, 319, 325, 328, 332, 333, 334, 335, 337, 339, 340, 341, 344, 345, 349, 350, 351, 354, 361, 365, 369, 370, 371, 372, 373, 374, 378, 384, 393, 395, 396, 398, 400, 402, 404, 406, 410, 428, 429, 441, 445, 446, 451, 453, 458, 465, 468, 475, 476, 477, 480, 481, 484, 488, 490, 495, 497, 498, 499, 503, 505, 516, 523, 525, 535, 536, 540, 541, 544, 545, 546, 548, 551, 556, 558, 561, 576, 581, 584, 587, 589, 591, 596, 599, 609, 610, 611, 615, 620, 626, 630, 631, 633, 636, 639, 645, 647, 650, 652, 656, 662, 665, 667, 672, 679, 680, 682, 685, 686, 691, 695, 696, 699, 702, 705, 711, 712, 714, 716, 717, 720, 725, 733, 735, 736, 737, 739, 740, 741, 742, 744, 746, 749, 750, 751, 753, 754, 755, 756, 757, 758, 759, 761, 762, 763, 768, 769, 770, 771, 772, 773, 774, 778, 779, 782, 783, 784, 785, 793, 794, 797, 799, 800, 801, 802, 803, 804, 805, 807, 808, 809, 810, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 825, 827, 828, 829, 830, 831, 832, 833, 834, 837, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 854, 856, 857, 858, 859, 860, 861, 863, 864, 865, 868, 869, 870, 871, 873, 875, 881, 883, 886, 889, 890, 891, 893, 896, 900, 903, 906, 908, 911, 912, 913, 914, 920, 934, 941, 942, 944, 947, 962, 966, 967, 971, 974, 975, 979, 982, 991, 996]
lose_34 =[[1, 2, 3, 4, 5, 6, 9, 11, 12, 15, 16, 18, 19, 20, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 36, 37, 38, 40, 41, 42, 225, 245, 803, 945]]
lose_292 = [1, 2, 5, 6, 7, 8, 9, 10, 17, 19, 46, 97, 112, 115, 116, 117, 122, 133, 134, 135, 137, 138, 139, 140, 141, 142, 145, 146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 224, 225, 226, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 242, 244, 245, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 262, 264, 265, 267, 268, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 283, 284, 288, 293, 296, 304, 308, 309, 314, 319, 327, 328, 329, 330, 332, 334, 339, 340, 342, 343, 346, 347, 348, 350, 355, 356, 357, 358, 361, 362, 365, 367, 368, 369, 370, 371, 372, 373, 374, 376, 377, 378, 379, 380, 382, 383, 384, 388, 389, 392, 393, 398, 403, 404, 405, 406, 408, 411, 413, 415, 416, 430, 449, 451, 464, 465, 472, 473, 474, 485, 514, 518, 531, 535, 536, 570, 583, 589, 599, 623, 626, 633, 644, 645, 652, 654, 656, 661, 662, 668, 691, 702, 704, 708, 714, 718, 719, 722, 724, 728, 731, 733, 734, 736, 737, 738, 739, 740, 741, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 755, 756, 757, 758, 761, 763, 764, 765, 766, 767, 768, 769, 770, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 785, 786, 787, 788, 789, 790, 791, 792, 797, 798, 800, 801, 802, 803, 807, 808, 810, 812, 813, 814, 816, 817, 818, 819, 820, 821, 823, 824, 825, 826, 827, 828, 830, 832, 833, 838, 839, 840, 841, 842, 845, 849, 854, 858, 867, 870, 874, 876, 878, 881, 885, 919, 950, 951, 952, 970]
lose_730 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 22, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 71, 72, 73, 75, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 119, 120, 121, 122, 124, 127, 131, 134, 137, 140, 142, 146, 151, 170, 176, 179, 181, 183, 184, 188, 189, 190, 191, 200, 205, 210, 214, 218, 219, 222, 224, 225, 229, 231, 236, 242, 245, 246, 250, 254, 257, 259, 260, 263, 265, 269, 272, 273, 277, 282, 285, 286, 288, 290, 291, 292, 294, 297, 302, 304, 312, 313, 315, 322, 323, 327, 328, 330, 331, 333, 335, 336, 339, 340, 342, 343, 345, 346, 348, 353, 355, 356, 358, 359, 361, 362, 364, 365, 368, 370, 371, 372, 374, 375, 376, 377, 379, 380, 381, 382, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 451, 452, 454, 456, 458, 459, 460, 463, 464, 466, 467, 470, 471, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 499, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 538, 539, 540, 541, 542, 543, 544, 545, 547, 548, 549, 550, 551, 552, 553, 554, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 776, 777, 778, 780, 781, 783, 784, 785, 787, 788, 790, 791, 792, 793, 794, 795, 796, 798, 799, 800, 801, 805, 807, 808, 809, 810, 812, 813, 814, 816, 817, 818, 819, 820, 821, 823, 824, 826, 827, 828, 829, 830, 831, 832, 835, 836, 837, 838, 840, 841, 843, 845, 846, 847, 848, 849, 851, 852, 854, 855, 857, 859, 860, 862, 863, 865, 866, 867, 870, 871, 872, 873, 874, 875, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 889, 890, 892, 894, 897, 898, 899, 900, 902, 903, 905, 907, 910, 911, 912, 913, 915, 916, 917, 918, 919, 920, 921, 922, 924, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 937, 938, 941, 942, 945, 946, 947, 949, 950, 953, 956, 957, 958, 959, 961, 966, 967, 968, 969, 970, 972, 973, 974, 975, 977, 978, 979, 980, 981, 982, 984, 986, 987, 988, 990, 991, 992, 994, 995, 996, 997, 998, 999]
lose_116 = [2, 4, 5, 7, 11, 12, 13, 15, 16, 19, 21, 26, 30, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 87, 88, 89, 90, 110, 147, 148, 149, 150, 154, 155, 180, 189, 190, 191, 192, 193, 194, 196, 198, 199, 200, 202, 207, 209, 215, 216, 223, 224, 233, 241, 257, 266, 272, 274, 281, 288, 290, 323, 324, 378, 415, 423, 424, 427, 428, 432, 439, 458, 460, 461, 462, 473, 509, 512, 826, 830, 841, 863, 936, 937]


lose_index = lose_571
# lose_index = lose_0
# mask_index = 0
# padded_masks = []
# handle = 0
# g_device = 0



# def mask_hook_get_information(module, input, output):
#     global padded_masks
#     n = output.shape[0]
#     c = output.shape[1]
#     w = output.shape[2]
#     h = output.shape[3]


#     basic_mask = np.ones(1000,dtype=bool)
#     basic_mask[lose_index] = 0
#     tmp = torch.Tensor(basic_mask)
#     tmp = tmp.repeat(100) #for yolov5 large input(large c h w)
#     one_d_mask = torch.repeat_interleave(tmp,150)
#     padding = (h*w - (len(one_d_mask)%(h*w))) % (h*w)
#     one_d_mask = torch.cat((one_d_mask,torch.ones(padding)))
#     print(len(one_d_mask))
#     padded_masks = one_d_mask.view(-1,w,h)
#     print(padded_masks.shape)

#     num_chunks = padded_masks.shape[0] // c
#     padded_masks = padded_masks[:num_chunks*c]
#     chunks = torch.split(padded_masks, c)
#     result = torch.stack(chunks)
#     padded_masks = result
#     handle.remove()

# def mask_hook_fill(module, input, output):

#     n = output.shape[0]
#     c = output.shape[1]
#     w = output.shape[2]
#     h = output.shape[3]
#     global mask_index
#     global padded_masks
#     if (c!=padded_masks.shape[0] or h!=padded_masks.shape[1] or w!=padded_masks.shape[2]):
#         basic_mask = np.ones(1000,dtype=bool)
#         basic_mask[lose_index] = 0
#         tmp = torch.Tensor(basic_mask)
#         tmp = tmp.repeat(100) #for yolov5 large input(large c h w)
#         one_d_mask = torch.repeat_interleave(tmp,150)
#         padding = (h*w - (len(one_d_mask)%(h*w))) % (h*w)
#         one_d_mask = torch.cat((one_d_mask,torch.ones(padding)))
#         print(len(one_d_mask))
#         padded_masks = one_d_mask.view(-1,w,h)
#         print(padded_masks.shape)

#         num_chunks = padded_masks.shape[0] // c
#         padded_masks = padded_masks[:num_chunks*c]
#         chunks = torch.split(padded_masks, c)
#         result = torch.stack(chunks)
#         padded_masks = result

#         output_mask = torch.ones(n,c,w,h).bool().to(g_device)
#         print('padded_masks',padded_masks.shape)
#         #output_mask = torch.ones(n,c,w,h).bool()

#         mask = padded_masks[mask_index % len (padded_masks)]
#         print(mask.shape)
#         mask.squeeze_(dim=0)
#         output_mask[:,:,:,:] = mask
#         output *= (output_mask)
#         mask_index = mask_index + 1
#     else:
#         output_mask = torch.ones(n,c,w,h).bool().to(g_device)
#         print('padded_masks',padded_masks.shape)
#         #output_mask = torch.ones(n,c,w,h).bool()

#         mask = padded_masks[mask_index % len (padded_masks)]
#         print(mask.shape)
#         mask.squeeze_(dim=0)
#         output_mask[:,:,:,:] = mask
#         output *= (output_mask)
        # mask_index = mask_index + 1
############################ hook #################cyj

mask_index_1 = 0
padded_masks_1 = []
handle_1 = 0

mask_index_2 = 0
padded_masks_2 = []
handle_2 = 0

g_device = 0

def mask_hook_get_information_1(module, input, output):
    global padded_masks_1
    n = output.shape[0]
    c = output.shape[1]
    w = output.shape[2]
    h = output.shape[3]


    basic_mask = np.ones(1000,dtype=bool)
    basic_mask[lose_index] = 0
    tmp = torch.Tensor(basic_mask)
    tmp = tmp.repeat(100) #for yolov5 large input(large c h w)
    one_d_mask = torch.repeat_interleave(tmp,150)
    padding = (h*w - (len(one_d_mask)%(h*w))) % (h*w)
    one_d_mask = torch.cat((one_d_mask,torch.ones(padding)))
    # print(len(one_d_mask))
    padded_masks_1 = one_d_mask.view(-1,w,h)
    # print(padded_masks.shape)

    num_chunks = padded_masks_1.shape[0] // c
    padded_masks_1 = padded_masks_1[:num_chunks*c]
    chunks = torch.split(padded_masks_1, c)
    result = torch.stack(chunks)
    padded_masks_1 = result
    handle_1.remove()

def mask_hook_fill_1(module, input, output):

    n = output.shape[0]
    c = output.shape[1]
    w = output.shape[2]
    h = output.shape[3]
    global mask_index_1
    global padded_masks_1
    if (c!=padded_masks_1.shape[0] or h!=padded_masks_1.shape[1] or w!=padded_masks_1.shape[2]):
        basic_mask = np.ones(1000,dtype=bool)
        basic_mask[lose_index] = 0
        tmp = torch.Tensor(basic_mask)
        tmp = tmp.repeat(100) #for yolov5 large input(large c h w)
        one_d_mask = torch.repeat_interleave(tmp,150)
        padding = (h*w - (len(one_d_mask)%(h*w))) % (h*w)
        one_d_mask = torch.cat((one_d_mask,torch.ones(padding)))
        # print(len(one_d_mask))
        padded_masks_1 = one_d_mask.view(-1,w,h)
        # print(padded_masks.shape)

        num_chunks = padded_masks_1.shape[0] // c
        padded_masks_1 = padded_masks_1[:num_chunks*c]
        chunks = torch.split(padded_masks_1, c)
        result = torch.stack(chunks)
        padded_masks_1 = result

        output_mask = torch.ones(n,c,w,h).bool().to(g_device)
        # print('padded_masks',padded_masks.shape)
        #output_mask = torch.ones(n,c,w,h).bool()

        mask = padded_masks_1[mask_index_1 % len (padded_masks_1)]
        # print(mask.shape)
        mask.squeeze_(dim=0)
        output_mask[:,:,:,:] = mask
        output *= (output_mask)
        mask_index_1 = mask_index_1 + 1
    else:
        output_mask = torch.ones(n,c,w,h).bool().to(g_device)
        # print('padded_masks',padded_masks.shape)
        #output_mask = torch.ones(n,c,w,h).bool()

        mask = padded_masks_1[mask_index_1 % len (padded_masks_1)]
        # print(mask.shape)
        mask.squeeze_(dim=0)
        output_mask[:,:,:,:] = mask
        output *= (output_mask)
        mask_index_1 = mask_index_1 + 1



def mask_hook_get_information_2(module, input, output):
    global padded_masks_2
    n = output.shape[0]
    c = output.shape[1]
    w = output.shape[2]
    h = output.shape[3]


    basic_mask = np.ones(1000,dtype=bool)
    basic_mask[lose_index] = 0
    tmp = torch.Tensor(basic_mask)
    tmp = tmp.repeat(100) #for yolov5 large input(large c h w)
    one_d_mask = torch.repeat_interleave(tmp,150)
    padding = (h*w - (len(one_d_mask)%(h*w))) % (h*w)
    one_d_mask = torch.cat((one_d_mask,torch.ones(padding)))
    # print(len(one_d_mask))
    padded_masks_2 = one_d_mask.view(-1,w,h)
    # print(padded_masks.shape)

    num_chunks = padded_masks_2.shape[0] // c
    padded_masks_2 = padded_masks_2[:num_chunks*c]
    chunks = torch.split(padded_masks_2, c)
    result = torch.stack(chunks)
    padded_masks_2 = result
    handle_2.remove()

def mask_hook_fill_2(module, input, output):

    n = output.shape[0]
    c = output.shape[1]
    w = output.shape[2]
    h = output.shape[3]
    global mask_index_2
    global padded_masks_2
    if (c!=padded_masks_2.shape[0] or h!=padded_masks_2.shape[1] or w!=padded_masks_2.shape[2]):
        basic_mask = np.ones(1000,dtype=bool)
        basic_mask[lose_index] = 0
        tmp = torch.Tensor(basic_mask)
        tmp = tmp.repeat(100) #for yolov5 large input(large c h w)
        one_d_mask = torch.repeat_interleave(tmp,150)
        padding = (h*w - (len(one_d_mask)%(h*w))) % (h*w)
        one_d_mask = torch.cat((one_d_mask,torch.ones(padding)))
        # print(len(one_d_mask))
        padded_masks_2 = one_d_mask.view(-1,w,h)
        # print(padded_masks.shape)

        num_chunks = padded_masks_2.shape[0] // c
        padded_masks_2 = padded_masks_2[:num_chunks*c]
        chunks = torch.split(padded_masks_2, c)
        result = torch.stack(chunks)
        padded_masks_2 = result

        output_mask = torch.ones(n,c,w,h).bool().to(g_device)
        # print('padded_masks',padded_masks.shape)
        #output_mask = torch.ones(n,c,w,h).bool()

        mask = padded_masks_2[mask_index_2 % len (padded_masks_2)]
        # print(mask.shape)
        mask.squeeze_(dim=0)
        output_mask[:,:,:,:] = mask
        output *= (output_mask)
        mask_index_2 = mask_index_2 + 1
    else:
        output_mask = torch.ones(n,c,w,h).bool().to(g_device)
        # print('padded_masks',padded_masks.shape)
        #output_mask = torch.ones(n,c,w,h).bool()

        mask = padded_masks_2[mask_index_2 % len (padded_masks_2)]
        # print(mask.shape)
        mask.squeeze_(dim=0)
        output_mask[:,:,:,:] = mask
        output *= (output_mask)
        mask_index_2 = mask_index_2 + 1





def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    print(model)
    # print(model.model.model[1])
    global handle_1
    global handle_2
    global g_device
    g_device = model.device
    handle_1 = model.model.model[1].register_forward_hook(mask_hook_get_information_1) #cyjnet
    model.model.model[1].register_forward_hook(mask_hook_fill_1)

    handle_2 = model.model.model[10].register_forward_hook(mask_hook_get_information_2) #cyjnet
    model.model.model[10].register_forward_hook(mask_hook_fill_2)


    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det)

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path('../datasets/coco/annotations/instances_val2017.json'))  # annotations
        pred_json = str(save_dir / f'{w}_predictions.json')  # predictions
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements('pycocotools>=2.0.6')
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.info('WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            subprocess.run(['zip', '-r', 'study.zip', 'study_*.txt'])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
