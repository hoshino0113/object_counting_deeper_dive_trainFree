import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
import argparse
import json
import numpy as np
import os
import copy
import time
import cv2
import random
from tqdm import tqdm
from os.path import exists,join
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import clip
from shi_segment_anything import sam_model_registry, SamPredictor
from shi_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from utils import *

# Define the calling syntax
parser = argparse.ArgumentParser(description="Counting with SAM")
parser.add_argument("-dp", "--data_path", type=str, default='./dataset/FSC147_384_V2/', help="Path to the FSC147 dataset")
parser.add_argument("-o", "--output_dir", type=str,default="./logsSave/FSC147", help="/Path/to/output/logs/")

#test--split doens't really matter since we can using any of the three for testing:
parser.add_argument("-ts", "--test-split", type=str, default='test', choices=["train", "test", "val"], help="what data split to evaluate on on")
parser.add_argument("-pt", "--prompt-type", type=str, default='box', choices=["box", "point", "text"], help="what type of information to prompt")
parser.add_argument("-d", "--device", type=str,default='cuda:0', help="device")
args = parser.parse_args()


# Define the helper function:
# jitter_box:
# For modifying the box supplied to the model
"""
Since in our FSC Dataset, all boxes are in form of [x1, x2, y1, y2], we can inject noises through
changing of these parameters. Width (W) and Height(H) are determined by the x and y.
"""
def jitter_box(box, max_shift=10, scale_jitter=0.2, scale_mode="both"):
    """
    box: [x1, y1, x2, y2]
    max_shift: max pixel shift in x or y
    scale_jitter: relative change in size, e.g. 0.2 = up to ±20%
    scale_mode: "both" | "enlarge" | "shrink" | "none"
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    # random translation
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)

    #random scale depending on mode
    if scale_mode == "none" or scale_jitter <= 0:
        s = 1.0
    elif scale_mode == "both":
        #shrink or enlarge: [1 - j, 1 + j]
        s = 1.0 + random.uniform(-scale_jitter, scale_jitter)
    elif scale_mode == "enlarge":
        # only scale up: [1, 1 + j]
        s = 1.0 + random.uniform(0.0, scale_jitter)
    elif scale_mode == "shrink":
        # only scale down: [1 - j, 1]
        s = 1.0 + random.uniform(-scale_jitter, 0.0)
    else:
        s = 1.0  # fallback

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    new_w = max(2, int(w * s))
    new_h = max(2, int(h * s))

    new_x1 = int(cx - new_w / 2) + dx
    new_y1 = int(cy - new_h / 2) + dy
    new_x2 = new_x1 + new_w
    new_y2 = new_y1 + new_h
    return [new_x1, new_y1, new_x2, new_y2]

#Define the helper function:
#jitter_point:
#For modifying the point supplied to the model
"""
Points are supplied by x and y.
"""
def jitter_point(pt, max_shift=10):
    """
    pt: [x, y]
    In this function, pt came from our clean_prompt, which contains the original x and y
    Now, we temper with it!
    """
    x, y = pt
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)
    return [x + dx, y + dy]

#Define the helper function:
#maybe_drop:
#For removing box/point arbitrary
def maybe_drop(prompts, drop_prob=0.0):
    """Randomly drop some prompts to simulate missing labels."""
    kept = []
    for p in prompts:
        if random.random() >= drop_prob:
            kept.append(p)
    #avoid empty prompts IF you want at least one
    if len(kept) == 0 and len(prompts) > 0:
        kept.append(random.choice(prompts))
    return kept

#Define the helper function:
#add_false_prompts
#randomly add box, point that do not contain the targeted object
def add_false_prompts(image_shape, num_false=0, box_mode=True):
    H, W = image_shape[:2]
    prompts = []
    for _ in range(num_false):
        if box_mode:
            x1 = random.randint(0, W - 10)
            y1 = random.randint(0, H - 10)
            x2 = random.randint(x1 + 5, min(W, x1 + 50))
            y2 = random.randint(y1 + 5, min(H, y1 + 50))
            prompts.append([x1, y1, x2, y2])
        else:
            x = random.randint(0, W - 1)
            y = random.randint(0, H - 1)
            prompts.append([x, y])
    return prompts

#Define the helper function:
#apply_point_mixed_noise
#mix the various point noise we defined earlier together
#All parameters are set inside this fucntion. Feel free to change them, but you must be warned
#that the more deviate the parameters are, the longer it takes to run !
def apply_point_mixed_noise(clean_prompts, image, level="mild"):
    """
    clean_prompts: list of [x, y]
    image: H x W x 3 (RGB)
    level: "mild" | "medium" | "extreme"
    """
    N = len(clean_prompts)

    #Set levels of severity
    if level == "mild":
        MAX_SHIFT = 3
        DROP_PROB = 0.10
        FALSE_RATIO = 0.10
    elif level == "medium":
        MAX_SHIFT = 6
        DROP_PROB = 0.30
        FALSE_RATIO = 0.20
    elif level == "extreme":
        MAX_SHIFT = 10
        DROP_PROB = 0.50
        FALSE_RATIO = 0.40
    else:
        # no noise
        return clean_prompts

    NUM_FALSE = max(1, int(FALSE_RATIO * max(1, N)))

    # 1) Drop some true points
    kept = []
    for p in clean_prompts:
        if random.random() >= DROP_PROB:
            kept.append(p)
    if len(kept) == 0 and len(clean_prompts) > 0:
        kept.append(random.choice(clean_prompts))

    #2) Shift remaining points
    noisy_true = [jitter_point(p, max_shift=MAX_SHIFT) for p in kept]

    #3) Add false background points
    false_pts = add_false_prompts(image.shape, num_false=NUM_FALSE, box_mode=False)

    # 4) Final mixed noisy prompts
    return noisy_true + false_pts


#Define the helper function:
#apply_box_mixed_noise
#mix the various box noise we defined earlier together
#All parameters are set inside this fucntion. Feel free to change them, but you must be warned
#that the more deviate the parameters are, the longer it takes to run !
def apply_box_mixed_noise(clean_boxes, image, level="mild"):
    """
    clean_boxes: list of [x1, y1, x2, y2]
    image: H x W x 3 (RGB)
    level: "mild" | "medium" | "extreme"
    """
    N = len(clean_boxes)

    if level == "mild":
        MAX_SHIFT = 3
        DROP_PROB = 0.10
        FALSE_RATIO = 0.10
    elif level == "medium":
        MAX_SHIFT = 6
        DROP_PROB = 0.30
        FALSE_RATIO = 0.20
    elif level == "extreme":
        MAX_SHIFT = 10
        DROP_PROB = 0.50
        FALSE_RATIO = 0.40
    else:
        return clean_boxes

    # 1) drop some boxes
    kept = maybe_drop(clean_boxes, drop_prob=DROP_PROB)

    # 2) shift ONLY (no scale)
    noisy_true = [
        jitter_box(b, max_shift=MAX_SHIFT, scale_jitter=0, scale_mode="none")
        for b in kept
    ]

    #3) add false boxes
    NUM_FALSE = max(1, int(FALSE_RATIO * max(1, N)))
    false_boxes = add_false_prompts(
        image.shape, num_false=NUM_FALSE, box_mode=True
    )

    return noisy_true + false_boxes




#############################################################
# For text prompt noise ONLY:
# It is for finding the 'upper'(or "superordinate" as we stated in the IEEE paper) catagory of an object
def find_group_for_class(cls_name, groups):
    """Return the group (set) that contains this cls_name, or None."""
    for g in groups:
        if cls_name in g:
            return g
    return None

def apply_text_noise(cls_name, mode="none", all_classes=None, groups=None):
    """
    cls_name: clean class name, e.g. "apples"
    mode: "none" | "mild" | "misspell" | "related" | "wrong"
    all_classes: set or list of all class names in dataset
    groups: list of sets, semantic groups (FRUITS, BIRDS, etc.)
    """
    if mode == "none":
        return cls_name

    #spelling noise
    def misspell(s):
        if len(s) <= 3:
            return s
        i = random.randint(0, len(s) - 1)
        if random.random() < 0.5:
            # drop one char
            return s[:i] + s[i+1:]
        else:
            # duplicate one char
            return s[:i] + s[i] + s[i:]
    
    if mode == "misspell":
        return misspell(cls_name)

    #mild descriptive noise
    if mode == "mild":
        prefixes = ["a", "the", "many", "several", "a group of"]
        prefix = random.choice(prefixes)
        return f"{prefix} {cls_name}"

    #related noise: choose another class from same semantic group 
    if mode == "related" and groups is not None:
        g = find_group_for_class(cls_name, groups)
        if g is not None:
            candidates = [c for c in g if c != cls_name]
            if candidates:
                return random.choice(candidates)
        #if not in any group, just use mild noise
        return f"{cls_name} on a table"

    #wrong noise: choose any other dataset class
    if mode == "wrong" and all_classes is not None:
        candidates = [c for c in all_classes if c != cls_name]
        if candidates:
            return random.choice(candidates)
    return cls_name


if __name__=="__main__": 

##############################Load all relavant files and folders
    data_path = args.data_path
    anno_file = data_path + 'annotation_FSC147_384.json'
    data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
    im_dir = data_path + 'images_384_VarV2' #Originally images, not work

    # Get all objects class in our dataset:
    with open(data_path+'ImageClasses_FSC147.txt') as f:
        class_lines = f.readlines()

    class_dict = {}
    for cline in class_lines:
        strings = cline.strip().split('\t')
        class_dict[strings[0]] = strings[1]
    
    #Build set of all class names from class_dict (image -> class name)
    all_classes = sorted(set(class_dict.values()))

    #Manually define a few semantic groups (feel free to add more!!)
    FRUITS = {
        "apples", "oranges", "grapes", "peaches", "strawberries",
        "tomatoes", "watermelon"
    }
    BIRDS = {
        "birds", "pigeons", "crows", "geese", "seagulls",
        "flamingos", "cranes"
    }
    BREADS = {
        "bread rolls", "buns", "baguette rolls", "naan bread"
    }
    NUTS_BEANS = {
        "nuts", "cashew nuts", "kidney beans", "red beans"
    }
    COFFEE_SNACK = {
        "coffee beans", "goldfish snack", "m&m pieces", "candy pieces"
    }
    GROUPS = [FRUITS, BIRDS, BREADS, NUTS_BEANS, COFFEE_SNACK]


    if not exists(args.output_dir):
        os.mkdir(args.output_dir)
        os.mkdir(args.output_dir+'/logs')
    
    if not exists(args.output_dir+'/%s'%args.test_split):
        os.mkdir(args.output_dir+'/%s'%args.test_split)

    if not exists(args.output_dir+'/%s/%s'%(args.test_split,args.prompt_type)):
        os.mkdir(args.output_dir+'/%s/%s'%(args.test_split,args.prompt_type))
    
    log_file = open(args.output_dir+'/logs/log-%s-%s.txt'%(args.test_split,args.prompt_type), "w")    

    with open(anno_file) as f:
        annotations = json.load(f)

    with open(data_split_file) as f:
        data_split = json.load(f)
##############################Load all relavant files and folders EOL

##############################Check if the prompt type is text, if so, invoke CLIP model
    if args.prompt_type=='text':
        from shi_segment_anything.automatic_mask_generator_text import SamAutomaticMaskGenerator
        clip_model, _ = clip.load("CS-ViT-B/16", device=args.device)
        clip_model.eval()
##############################Check if the prompt type is text, if so, invoke CLIP model EOL

##############################Load the vanilla SAM model, and use it to create a custom SAM for masks generation
    sam_checkpoint = "./pretrain/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=args.device)
    
    mask_generator = SamAutomaticMaskGenerator(model=sam)
##############################Load the vanilla SAM model, and use it to create a custom SAM for masks generation EOL

##############################

# In this part, we:
# 1. Initialize four evaluation matrices: 
#     - MAE: How many objects on average is the model off by (mean absolute err)
#     - RMSE: How many times the large errors occur? (root mean squared err), similar to 1% low fps in gaming industry
#     - NAE: How large the error compares with the ground truth? (normalized absolute err)
#     - SRE: Further tells us how far off those outliers are (squared relative err, or Relative Squared Error (RSE))
#     The lower the ALL four of these benchmarks are, the better the result is
#     https://accessibleai.dev/post/regression_metrics/
# 2. Only interate through the first 20 images due to hardware restrictions
# 3. For each iteration, we call the custom SAM to generate the masks, with three priors supplied in the shi_segment_anything folder
# 4. Each iteration will accumulate the four evalutation matrices
# 5. Once the iterations are done, we output the average of each matrices and output the result

    MAE = 0
    RMSE = 0
    NAE = 0
    SRE = 0
    wrong_mode = 0

    im_ids = data_split[args.test_split] #DOes not matter since we can use any one of test, train, val dataset

    SUBSET_FILE = data_path + "fsc147_test_100_stratified.txt"  # use None to disable
    if SUBSET_FILE is not None:
        with open(SUBSET_FILE) as f:
            selected_ids = set(line.strip() for line in f)
        im_ids = [img for img in im_ids if img in selected_ids]
        print(f"Using subset: {len(im_ids)} images")

    #im_ids = im_ids[:100]  # only use the first 100 images (Old code preserved for reference, before randomization)
    for i,im_id in tqdm(enumerate(im_ids)):
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates'] #supply the box examplar file for later use (later if the prompt type is 'box')
        dots = np.array(anno['points']) #get the ground truth
        
        #call standard cv lib for importaing images
        image = cv2.imread('{}/{}'.format(im_dir, im_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #if prompt type is 'text', invoke the CLIP model to create 'box' for the SAM
        if args.prompt_type=='text':
            cls_clean = class_dict[im_id]

            # Choose text noise level here
            TEXT_NOISE_MODE = "wrong"   # "none" | "mild" | "misspell" | "related" | "wrong"
            if TEXT_NOISE_MODE == "wrong":
                wrong_mode = 1

            cls_noisy = apply_text_noise(
                cls_clean,
                mode=TEXT_NOISE_MODE,
                all_classes=all_classes,
                groups=GROUPS
            )

            # For debugging ONLY, do not enable when not
            # print(f"{im_id}: clean='{cls_clean}' noisy='{cls_noisy}'")

            input_prompt = get_clip_bboxs(clip_model, image, cls_noisy, args.device)

        else:
            clean_prompts = []
            for bbox in bboxes:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[2][0]
                y2 = bbox[2][1]

                if args.prompt_type == 'box':
                    clean_prompts.append([x1, y1, x2, y2])
                elif args.prompt_type == 'point':
                    clean_prompts.append([(x1 + x2) // 2, (y1 + y2) // 2])

            #######NOISE CONFIG#########
            NOISE_MODE = "none"      # "none" | "shift_box" | "shift_point" | "drop" | "add_false"
                                          # "point_mix_mild" | "point_mix_medium" | "point_mix_extreme"
                                          # "box_mix_mild"   | "box_mix_medium"   | "box_mix_extreme"
            MAX_SHIFT = 0                # pixels
            SCALE_JITTER = 0           # e.g. 0.2 = up to ±20% size change
            BOX_SCALE_MODE = "none"       # "both" | "enlarge" | "shrink" | "none"
            DROP_PROB = 0                 # 30% boxes/points missing
            FALSE_RATIO = 0                # false background prompts
            ############################

            input_prompt = clean_prompts

            if NOISE_MODE == "shift_box" and args.prompt_type == "box":
                input_prompt = [
                    jitter_box(
                        b,
                        max_shift=MAX_SHIFT,
                        scale_jitter=SCALE_JITTER,
                        scale_mode=BOX_SCALE_MODE
                    )
                    for b in clean_prompts
                ]
            elif NOISE_MODE == "shift_point" and args.prompt_type == "point":
                input_prompt = [jitter_point(p, max_shift=MAX_SHIFT)
                                for p in clean_prompts]
            elif NOISE_MODE == "drop":
                input_prompt = maybe_drop(clean_prompts, drop_prob=DROP_PROB)
            elif NOISE_MODE == "add_false":
                N = len(clean_prompts)
                NUM_FALSE = max(1, int(FALSE_RATIO * max(1, N)))

                box_mode = (args.prompt_type == "box")

                input_prompt = list(clean_prompts)
                input_prompt += add_false_prompts(
                    image.shape,
                    num_false=NUM_FALSE,
                    box_mode=box_mode
                )
            elif NOISE_MODE in ["point_mix_mild", "point_mix_medium", "point_mix_extreme"]:
                level = NOISE_MODE.replace("point_mix_", "")  # "mild"/"medium"/"extreme"
                input_prompt = apply_point_mixed_noise(clean_prompts, image, level=level)

            elif NOISE_MODE in ["box_mix_mild", "box_mix_medium", "box_mix_extreme"]:
                level = NOISE_MODE.replace("box_mix_", "")  # "mild"/"medium"/"extreme"
                input_prompt = apply_box_mixed_noise(clean_prompts, image, level=level)

            else: 
                input_prompt = clean_prompts

        
        # create masks by invoking the mask generator we created earlier
        #a) in order to add noise, we need to change the input_prompt before calling mask_generator
        # b) if we want to incorporating texture/shape cues such as SIFT/HOG/LoG we've learned in class,
        # need to filter the masks it genrated after generator was called
        masks = mask_generator.generate(image, input_prompt)

        if wrong_mode != 1:
            gt_cnt = dots.shape[0]
        else:
            gt_cnt = 0.001

        pred_cnt = len(masks)

        #hmm, and maybe we can do something to make the evaluation more accurate, by changing the way the
        # model accumulate the errors
        print(pred_cnt, gt_cnt, abs(pred_cnt-gt_cnt))
        log_file.write("%d: %d,%d,%d\n"%(i, pred_cnt, gt_cnt,abs(pred_cnt-gt_cnt)))
        log_file.flush()

        err = abs(gt_cnt - pred_cnt)
        MAE = MAE + err
        RMSE = RMSE + err**2
        NAE = NAE+err/gt_cnt
        SRE = SRE+err**2/gt_cnt

        #Mask visualization
        
        # fig = plt.figure()
        # plt.axis('off')
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        # ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        # plt.imshow(image)
        # show_anns(masks, plt.gca())
        # plt.savefig('%s/%s/%03d_mask.png'%(args.output_dir,args.test_split,i), bbox_inches='tight', pad_inches=0)
        # plt.close()

    MAE = MAE/len(im_ids)
    RMSE = math.sqrt(RMSE/len(im_ids))
    NAE = NAE/len(im_ids)
    SRE = math.sqrt(SRE/len(im_ids))

    print("MAE:%0.2f,RMSE:%0.2f,NAE:%0.2f,SRE:%0.2f"%(MAE,RMSE,NAE,SRE))
    log_file.write("MAE:%0.2f,RMSE:%0.2f,NAE:%0.2f,SRE:%0.2f"%(MAE,RMSE,NAE,SRE))
    log_file.close()

        
