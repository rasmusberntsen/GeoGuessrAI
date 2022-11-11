class CustomImageDataset():
    #Making init
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file).iloc[:, 1:3] #Reading the annotations
        self.img_dir = img_dir #Getting the image directory
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels) #Returning number of samples in dataset
    
    #The getitem function is used to get an image and a label at a given index and resizes the image to the desired dimension
    # Using the read_image function we convert the image to a tensor.
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1]) #Specifying the path to the image
        image = read_image(img_path) #Reading the image
        label = self.img_labels.iloc[idx, 0] #Getting the label for the imange
        if self.transform:
            image = self.transform(image) #Transforming the image
        if self.target_transform:
            label = self.target_transform(label) #Transforming the label
        return image, label