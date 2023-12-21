import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from torch import nn, optim

from models import *

DATASET_PATH = './data'
SAVE_PATH = './data/working/'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#noise


lose_571 = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 22, 23, 25, 27, 29, 30, 34, 35, 36, 37, 38, 39, 41, 43, 44, 45, 46, 47, 48, 50, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 65, 66, 67, 68, 69, 71, 72, 74, 75, 76, 77, 79, 80, 82, 84, 88, 93, 96, 97, 99, 104, 105, 106, 109, 110, 111, 112, 114, 115, 116, 120, 122, 126, 130, 131, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 145, 146, 147, 149, 150, 152, 153, 154, 155, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 177, 178, 179, 180, 181, 182, 183, 185, 187, 188, 189, 190, 191, 193, 194, 195, 196, 198, 200, 201, 202, 203, 204, 205, 207, 208, 209, 210, 211, 212, 215, 216, 218, 219, 221, 224, 243, 246, 249, 257, 258, 265, 269, 277, 280, 282, 286, 288, 290, 291, 292, 297, 299, 303, 309, 312, 313, 317, 320, 322, 328, 330, 333, 334, 335, 338, 339, 341, 342, 343, 344, 350, 352, 353, 355, 357, 358, 362, 364, 365, 369, 370, 374, 378, 388, 391, 396, 397, 398, 399, 403, 404, 407, 410, 417, 422, 425, 428, 430, 434, 435, 437, 439, 440, 442, 444, 446, 448, 452, 459, 460, 461, 462, 465, 466, 468, 469, 471, 472, 473, 474, 480, 481, 482, 483, 491, 492, 498, 509, 511, 515, 520, 521, 522, 524, 525, 526, 527, 528, 529, 530, 531, 532, 534, 535, 536, 537, 538, 539, 541, 543, 544, 545, 546, 547, 548, 550, 551, 552, 555, 556, 558, 560, 561, 563, 564, 565, 566, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 580, 581, 582, 583, 584, 585, 586, 587, 588, 590, 591, 592, 594, 595, 596, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 614, 615, 617, 618, 619, 620, 621, 629, 631, 635, 639, 641, 644, 646, 650, 653, 654, 655, 657, 659, 664, 665, 670, 672, 675, 676, 677, 679, 680, 681, 683, 684, 685, 686, 690, 692, 693, 694, 695, 696, 697, 698, 701, 704, 706, 709, 713, 715, 716, 718, 719, 720, 724, 731, 733, 734, 735, 736, 740, 741, 745, 749, 750, 756, 758, 763, 764, 766, 770, 772, 773, 785, 786, 788, 789, 790, 791, 795, 797, 798, 801, 802, 805, 806, 809, 811, 815, 817, 819, 820, 822, 823, 824, 825, 828, 831, 832, 833, 834, 835, 836, 837, 839, 840, 841, 842, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 933, 934, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 947, 948, 952, 953, 954, 955, 958, 959, 960, 961, 962, 963, 964, 965, 967, 968, 970, 971, 972, 976, 978, 979, 980, 981, 982, 983, 984, 985, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998]
#lose_0_1 = torch.arange(1,1000,2)
lose_0 = []
lose_406 =[0, 3, 7, 8, 11, 12, 13, 16, 18, 32, 33, 35, 36, 40, 42, 43, 44, 45, 46, 48, 49, 50, 52, 59, 64, 67, 71, 74, 79, 84, 88, 90, 97, 100, 104, 108, 110, 116, 117, 119, 121, 123, 127, 128, 129, 131, 136, 140, 144, 147, 150, 152, 153, 154, 155, 156, 157, 158, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 172, 173, 174, 175, 176, 177, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 198, 199, 200, 201, 202, 204, 206, 207, 210, 211, 212, 213, 214, 215, 216, 218, 220, 221, 223, 224, 225, 226, 227, 228, 229, 232, 233, 234, 235, 236, 238, 239, 241, 242, 243, 246, 250, 251, 252, 253, 254, 260, 263, 264, 265, 266, 267, 268, 270, 271, 273, 275, 276, 277, 278, 281, 282, 283, 284, 292, 296, 300, 301, 307, 309, 314, 317, 319, 325, 328, 332, 333, 334, 335, 337, 339, 340, 341, 344, 345, 349, 350, 351, 354, 361, 365, 369, 370, 371, 372, 373, 374, 378, 384, 393, 395, 396, 398, 400, 402, 404, 406, 410, 428, 429, 441, 445, 446, 451, 453, 458, 465, 468, 475, 476, 477, 480, 481, 484, 488, 490, 495, 497, 498, 499, 503, 505, 516, 523, 525, 535, 536, 540, 541, 544, 545, 546, 548, 551, 556, 558, 561, 576, 581, 584, 587, 589, 591, 596, 599, 609, 610, 611, 615, 620, 626, 630, 631, 633, 636, 639, 645, 647, 650, 652, 656, 662, 665, 667, 672, 679, 680, 682, 685, 686, 691, 695, 696, 699, 702, 705, 711, 712, 714, 716, 717, 720, 725, 733, 735, 736, 737, 739, 740, 741, 742, 744, 746, 749, 750, 751, 753, 754, 755, 756, 757, 758, 759, 761, 762, 763, 768, 769, 770, 771, 772, 773, 774, 778, 779, 782, 783, 784, 785, 793, 794, 797, 799, 800, 801, 802, 803, 804, 805, 807, 808, 809, 810, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 825, 827, 828, 829, 830, 831, 832, 833, 834, 837, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 854, 856, 857, 858, 859, 860, 861, 863, 864, 865, 868, 869, 870, 871, 873, 875, 881, 883, 886, 889, 890, 891, 893, 896, 900, 903, 906, 908, 911, 912, 913, 914, 920, 934, 941, 942, 944, 947, 962, 966, 967, 971, 974, 975, 979, 982, 991, 996]
lose_34 =[[1, 2, 3, 4, 5, 6, 9, 11, 12, 15, 16, 18, 19, 20, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 36, 37, 38, 40, 41, 42, 225, 245, 803, 945]]
lose_292 = [1, 2, 5, 6, 7, 8, 9, 10, 17, 19, 46, 97, 112, 115, 116, 117, 122, 133, 134, 135, 137, 138, 139, 140, 141, 142, 145, 146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 224, 225, 226, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 242, 244, 245, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 262, 264, 265, 267, 268, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 283, 284, 288, 293, 296, 304, 308, 309, 314, 319, 327, 328, 329, 330, 332, 334, 339, 340, 342, 343, 346, 347, 348, 350, 355, 356, 357, 358, 361, 362, 365, 367, 368, 369, 370, 371, 372, 373, 374, 376, 377, 378, 379, 380, 382, 383, 384, 388, 389, 392, 393, 398, 403, 404, 405, 406, 408, 411, 413, 415, 416, 430, 449, 451, 464, 465, 472, 473, 474, 485, 514, 518, 531, 535, 536, 570, 583, 589, 599, 623, 626, 633, 644, 645, 652, 654, 656, 661, 662, 668, 691, 702, 704, 708, 714, 718, 719, 722, 724, 728, 731, 733, 734, 736, 737, 738, 739, 740, 741, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 755, 756, 757, 758, 761, 763, 764, 765, 766, 767, 768, 769, 770, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 785, 786, 787, 788, 789, 790, 791, 792, 797, 798, 800, 801, 802, 803, 807, 808, 810, 812, 813, 814, 816, 817, 818, 819, 820, 821, 823, 824, 825, 826, 827, 828, 830, 832, 833, 838, 839, 840, 841, 842, 845, 849, 854, 858, 867, 870, 874, 876, 878, 881, 885, 919, 950, 951, 952, 970]
lose_730 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 22, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 71, 72, 73, 75, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 119, 120, 121, 122, 124, 127, 131, 134, 137, 140, 142, 146, 151, 170, 176, 179, 181, 183, 184, 188, 189, 190, 191, 200, 205, 210, 214, 218, 219, 222, 224, 225, 229, 231, 236, 242, 245, 246, 250, 254, 257, 259, 260, 263, 265, 269, 272, 273, 277, 282, 285, 286, 288, 290, 291, 292, 294, 297, 302, 304, 312, 313, 315, 322, 323, 327, 328, 330, 331, 333, 335, 336, 339, 340, 342, 343, 345, 346, 348, 353, 355, 356, 358, 359, 361, 362, 364, 365, 368, 370, 371, 372, 374, 375, 376, 377, 379, 380, 381, 382, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 451, 452, 454, 456, 458, 459, 460, 463, 464, 466, 467, 470, 471, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 499, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 538, 539, 540, 541, 542, 543, 544, 545, 547, 548, 549, 550, 551, 552, 553, 554, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 776, 777, 778, 780, 781, 783, 784, 785, 787, 788, 790, 791, 792, 793, 794, 795, 796, 798, 799, 800, 801, 805, 807, 808, 809, 810, 812, 813, 814, 816, 817, 818, 819, 820, 821, 823, 824, 826, 827, 828, 829, 830, 831, 832, 835, 836, 837, 838, 840, 841, 843, 845, 846, 847, 848, 849, 851, 852, 854, 855, 857, 859, 860, 862, 863, 865, 866, 867, 870, 871, 872, 873, 874, 875, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 889, 890, 892, 894, 897, 898, 899, 900, 902, 903, 905, 907, 910, 911, 912, 913, 915, 916, 917, 918, 919, 920, 921, 922, 924, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 937, 938, 941, 942, 945, 946, 947, 949, 950, 953, 956, 957, 958, 959, 961, 966, 967, 968, 969, 970, 972, 973, 974, 975, 977, 978, 979, 980, 981, 982, 984, 986, 987, 988, 990, 991, 992, 994, 995, 996, 997, 998, 999]
lose_116 = [2, 4, 5, 7, 11, 12, 13, 15, 16, 19, 21, 26, 30, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 87, 88, 89, 90, 110, 147, 148, 149, 150, 154, 155, 180, 189, 190, 191, 192, 193, 194, 196, 198, 199, 200, 202, 207, 209, 215, 216, 223, 224, 233, 241, 257, 266, 272, 274, 281, 288, 290, 323, 324, 378, 415, 423, 424, 427, 428, 432, 439, 458, 460, 461, 462, 473, 509, 512, 826, 830, 841, 863, 936, 937]

lose_index = lose_571


mask_index_1 = 0
padded_masks_1 = []
handle_1 = 0

mask_index_2 = 0
padded_masks_2 = []
handle_2 = 0


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

        output_mask = torch.ones(n,c,w,h).bool().to(device)
        # print('padded_masks',padded_masks.shape)
        #output_mask = torch.ones(n,c,w,h).bool()

        mask = padded_masks_1[mask_index_1 % len (padded_masks_1)]
        # print(mask.shape)
        mask.squeeze_(dim=0)
        output_mask[:,:,:,:] = mask
        output *= (output_mask)
        mask_index_1 = mask_index_1 + 1
    else:
        output_mask = torch.ones(n,c,w,h).bool().to(device)
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

        output_mask = torch.ones(n,c,w,h).bool().to(device)
        # print('padded_masks',padded_masks.shape)
        #output_mask = torch.ones(n,c,w,h).bool()

        mask = padded_masks_2[mask_index_2 % len (padded_masks_2)]
        # print(mask.shape)
        mask.squeeze_(dim=0)
        output_mask[:,:,:,:] = mask
        output *= (output_mask)
        mask_index_2 = mask_index_2 + 1
    else:
        output_mask = torch.ones(n,c,w,h).bool().to(device)
        # print('padded_masks',padded_masks.shape)
        #output_mask = torch.ones(n,c,w,h).bool()

        mask = padded_masks_2[mask_index_2 % len (padded_masks_2)]
        # print(mask.shape)
        mask.squeeze_(dim=0)
        output_mask[:,:,:,:] = mask
        output *= (output_mask)
        mask_index_2 = mask_index_2 + 1


#-------------------------------------
# def mask_hook_get_information(module, input, output):
#     global padded_masks
#     n = output.shape[0]
#     c = output.shape[1]
#     h = output.shape[2]
#     w = output.shape[3]

#     feat_size = w
#     basic_mask = np.ones(1000,dtype=bool)
#     basic_mask[lose_571] = 0
#     tmp = torch.Tensor(basic_mask)
#     one_d_mask = torch.repeat_interleave(tmp,150)
#     padding = (feat_size*feat_size - (len(one_d_mask)%(feat_size*feat_size))) % (feat_size*feat_size)
#     one_d_mask = torch.cat((one_d_mask,torch.ones(padding)))
#     padded_masks = one_d_mask.view(-1,feat_size,feat_size)

#     num_chunks = padded_masks.shape[0] // c
#     padded_masks = padded_masks[:num_chunks*c]
#     chunks = torch.split(padded_masks, c)
#     result = torch.stack(chunks)
#     padded_masks = result
#     handle.remove()

# def mask_hook_fill(module, input, output):

#     n = output.shape[0]
#     c = output.shape[1]
#     h = output.shape[2]
#     w = output.shape[3]

#     output_mask = torch.ones(n,c,w,h).bool().to(device)
#     global mask_index
#     mask = padded_masks[mask_index % len (padded_masks)]
#     output_mask[:,:,:,:] = mask
#     output *= (output_mask)
#     mask_index = mask_index + 1



# 定义数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# 加载测试集
testset = torchvision.datasets.CIFAR10(root= DATASET_PATH, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=2)

# 定义类别标签
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = MobileNetV2()
#net = ResNet50()
model_path = './checkpoint/ckpt_test_mobile.pth'

# model_path = './checkpoint/ckpt_test_mobile.pth'
#model_path = './checkpoint/ckpt_resnet18_base.pth'

#model_path= './checkpoint/ckpt_res18_layer1_layer3__dropout.pth'

model_data = torch.load(model_path)
weights_dict = {}
for k, v in model_data['net'].items():
    new_k = k.replace('module.', '') if 'module' in k else k
    weights_dict[new_k] = v

net.load_state_dict(weights_dict)
net = net.to(device)
print(net)

#add noise

# mobilenet
handle_1 = net.layers[3].register_forward_hook(mask_hook_get_information_1)
net.layers[3].register_forward_hook(mask_hook_fill_1)

handle_2= net.layers[9].register_forward_hook(mask_hook_get_information_2)
net.layers[9].register_forward_hook(mask_hook_fill_2)

#resnet
# handle_1 = net.layer1.register_forward_hook(mask_hook_get_information_1)
# net.layer1.register_forward_hook(mask_hook_fill_1)

# handle_2 = net.layer3.register_forward_hook(mask_hook_get_information_2)
# net.layer3.register_forward_hook(mask_hook_fill_2)

# handle = net.layer3[1].conv3.register_forward_hook(mask_hook_get_information)
# net.layer3[1].conv3.register_forward_hook(mask_hook_fill)



# 设置模型为评估模式
net = net.eval()

# 定义正确率和总数变量
correct = 0
total = 0

# 遍历测试集数据
with torch.no_grad(): # 不计算梯度，节省内存和时间
    for (data, label) in testloader:
        images, labels = data.to(device), label.to(device) # 获取一批图像和标签
        outputs = net(images) # 获取模型输出
        _, predicted = torch.max(outputs.data, 1) # 获取预测类别
        total += labels.size(0) # 更新总数
        correct += (predicted == labels).sum().item() # 更新正确数

# 打印测试集上的准确率
print('Accuracy of the network on the 10000 test images: %.2f %%' % (
    100 * correct / total))
