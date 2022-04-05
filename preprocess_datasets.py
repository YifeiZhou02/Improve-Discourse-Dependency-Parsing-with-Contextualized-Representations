from EDU import EDU
import os, json, re
import pickle
import argparse
def get_sci_edus(filepath):
    """
    load each sciedu
    """
    with open(filepath, 'r') as fb:
        train = json.loads(fb.read().encode('utf-8'))['root']
    EDUs = []
    sentenceNo = 1
    sentenceID = 1
    for edu_dict in train:
        if edu_dict['id'] == 0:
            continue
        EDUs.append(EDU([edu_dict['id'], edu_dict['parent'], edu_dict['relation'],
                    re.sub('<S>|\r' ,'',edu_dict['text']), sentenceNo, sentenceID],
                   [1]))
        if '<S>' in edu_dict['text']:
            sentenceNo += 1
            sentenceID = 1
        else:
            sentenceID += 1
    return EDUs

def get_edus(filepath):
    with open(filepath, 'r') as fb:
        train = json.loads(fb.read().encode('utf-8'))['root']
    EDUs = []
    sentenceNo = 1
    sentenceID = 1
    for edu_dict in train:
        if edu_dict['id'] == 0:
            continue
        EDUs.append(EDU([edu_dict['id'], edu_dict['parent'], edu_dict['relation'],
                    re.sub('<S>|\r' ,'',edu_dict['text']), sentenceNo, sentenceID],
                   [1]))
        if '<S>' in edu_dict['text'] or '。' in edu_dict['text']:
            sentenceNo += 1
            sentenceID = 1
        else:
            sentenceID += 1
    return EDUs

def process_cdtb(filepath):
    """
    load the cdtb dataset into the edu format
    """
    data = []
    train_prefix = filepath+'/train/'
    for fp in os.listdir(filepath+'/train'):
        if not ".dep" in fp:
            continue
        data.append(get_edus(train_prefix+fp))
    dev_data = []
    dev_prefix = filepath+'/dev/'
    for fp in os.listdir(filepath+'/dev'):
        if not ".dep" in fp:
            continue
        dev_data.append(get_edus(dev_prefix+fp))
    test_data = []
    test_prefix = filepath+'/test/'
    for fp in os.listdir(filepath+'/test'):
        if not ".dep" in fp:
            continue
        test_data.append(get_edus(test_prefix+fp))
    return data, dev_data, test_data


def process_scidtb(filepath):
    """
    load the scidtb dataset into the edu format
    """
    data = []
    train_prefix = filepath+'/train/'
    for fp in os.listdir(filepath+'/train'):
        data.append(get_edus(train_prefix+fp))
    dev_data = []
    dev_prefix = filepath+'/dev/gold/'
    for fp in os.listdir(filepath+'/dev/gold'):
        dev_data.append(get_edus(dev_prefix+fp))
    test_data = []
    test_prefix = filepath+'/test/gold/'
    for fp in os.listdir(filepath+'/test/gold'):
        test_data.append(get_edus(test_prefix+fp))
    return data, dev_data, test_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scidtb", type=str)
    parser.add_argument("--path", default=".", type=str)
    parser.add_argument("--target_dir", default="", type=str)
    args = parser.parse_args()
    target_path = args.target_dir
    if args.dataset == "scidtb":
        data, dev_data, test_data = process_scidtb(args.path)
        with open(target_path + 'sci.data', 'wb') as fb:
            pickle.dump(data, fb)
        with open(target_path + 'sci_dev.data', 'wb') as fb:
            pickle.dump(dev_data, fb)
        with open(target_path + 'sci_test.data', 'wb') as fb:
            pickle.dump(test_data, fb)
    if args.dataset == "cdtb":
        data, dev_data, test_data = process_cdtb(args.path)
        with open(target_path + 'cdtb.data', 'wb') as fb:
            pickle.dump(data, fb)
        with open(target_path + 'cdtb_dev.data', 'wb') as fb:
            pickle.dump(dev_data, fb)
        with open(target_path + 'cdtb_test.data', 'wb') as fb:
            pickle.dump(test_data, fb)

if __name__ == "__main__":
    main()
