with open('/home/sakurai/git2/lab2021/experiments/result/token_classification/version_43/pred_sentence.txt', 'r') as f:
    file_data = f.readlines()
    for line in file_data:
        if line != "\n":
            print(line)