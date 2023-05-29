import argparse
import numpy as np
import os
import onnxruntime
from PIL import Image
import torchvision.transforms as transforms

def main(args):
    # Load ONNX model
    sess = onnxruntime.InferenceSession(args.load_ckpt)

    # Process images
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load and process a single image
    image_path = os.path.join(args.video_path, 'vertical.jpg')
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)

    # Convert tensor to numpy array
    input_array = image.cpu().detach().numpy()

    y_hat = sess.run(None, {sess.get_inputs()[0].name: input_array})
    # print("Type of y_hat:", type(y_hat))
    # print("Length of y_hat:", len(y_hat))
    # print("y_hat content:", y_hat)

    # Assuming y_hat is a list of one output tensor, convert it to a NumPy array
    # y_hat_array = np.array(y_hat[0])

    # Then you can call transpose on the array
    # y_hat = np.transpose(y_hat_array, (0, 2, 3, 1)) 

    # # Run the model
    # y_hat = sess.run(None, {sess.get_inputs()[0].name: input_array})
    # print("Type of y_hat:", type(y_hat))
    # print("Length of y_hat:", len(y_hat))
    # print("y_hat content:", y_hat)


    # # Back to image scale (0-255)
    # y_hat = np.array(y_hat)
    # y_hat = np.clip(y_hat, -1.0, 1.0)
    # y_hat = ((y_hat + 1.0) * 127.5).astype(np.uint8)
    # print(type(y_hat), len(y_hat), y_hat)

    # Assuming y_hat is a list of one output tensor, convert it to a NumPy array
    y_hat_array = np.array(y_hat[0])

    # Then you can call transpose on the array
    y_hat = np.transpose(y_hat_array, (0, 2, 3, 1))

    # Clip the values between -1 and 1
    y_hat = np.clip(y_hat, -1.0, 1.0)

    # Scale the values between 0 and 255
    y_hat = ((y_hat + 1.0) * 127.5)

    # Cast to uint8
    y_hat = y_hat.astype(np.uint8)

    # Now y_hat should be a valid input for Image.fromarray
    im = Image.fromarray(np.squeeze(y_hat))



    # Save the result
    fn = os.path.join(args.save_dir, 'STBVMM_%s.png' % (args.mode))
    im = Image.fromarray(np.squeeze(y_hat))
    im.save(fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Swin Transformer Based Video Motion Magnification')

    # Application parameters
    parser.add_argument('-i', '--video_path', type=str, metavar='PATH', required=True,
                        help='path to the directory containing the image')
    parser.add_argument('-c', '--load_ckpt', type=str, metavar='PATH', required=True,
                        help='path to load ONNX model')
    parser.add_argument('-o', '--save_dir', default='demo', type=str, metavar='PATH',
                        help='path to save generated frames (default: demo)')
    parser.add_argument('--mode', default='static', type=str, choices=['static', 'dynamic'],
                        help='magnification mode (static, dynamic)')

    args = parser.parse_args()
    main(args)
