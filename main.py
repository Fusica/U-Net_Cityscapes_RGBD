import os

if __name__ == "__main__":
    if not os.path.exists('model.pth'):
        print("Training the model...")
        os.system('python train.py')
    print("Testing the model...")
    os.system('python test.py')
