// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OpenCvSharp;
using TorchSharp;
using TorchSharp.torchvision;
using static TorchSharp.torch;

namespace AutoMLExample
{
    /// <summary>
    /// Data reader utility for datasets that follow the MNIST data set's layout:
    ///
    /// A number of single-channel (grayscale) images are laid out in a flat file with four 32-bit integers at the head.
    /// The format is documented at the bottom of the page at: http://yann.lecun.com/exdb/mnist/
    /// </summary>
    public sealed class DataReader : IDisposable
    {
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="path">Path to the folder containing the image files.</param>
        /// <param name="prefix">The file name prefix, either 'train' or 't10k' (the latter being the test data set).</param>
        /// <param name="batch_size">The batch size</param>
        /// <param name="shuffle">Randomly shuffle the images.</param>
        /// <param name="device">The device, i.e. CPU or GPU to place the output tensors on.</param>
        /// <param name="reverseGBR"></param>
        /// <param name="augmentation"></param>
        public DataReader(string path, string prefix, int batch_size = 32, bool shuffle = false, torch.Device device = null, bool reverseGBR = true, bool augmentation = false)
        {
            // The MNIST data set is small enough to fit in memory, so let's load it there.
            BatchSize = batch_size;
            reverse = reverseGBR;
            this.shuffle = shuffle;
            augment = augmentation;

            var workerPath = Path.Combine(path, prefix);

            var count = -1;
            this.device = device;

            var directories = new DirectoryInfo(workerPath).GetDirectories();

            int current_label = 0;
            foreach (var folder in directories)
            {
                //var label = int.Parse(folder.FullName);
                var label = current_label++;

                foreach (var img_file_name in folder.GetFiles())
                {
                    var imgLocation = Path.Combine(path, prefix, folder.FullName, img_file_name.FullName);
                    imgLocationList.Add(imgLocation);
                    labelList.Add(label);
                    count++;
                }
            }
            // Set up the indices array.
            Random rnd = new Random();
            indices = !shuffle ?
                Enumerable.Range(0, count).ToArray() :
                Enumerable.Range(0, count).OrderBy(c => rnd.Next()).ToArray();

            batchCount = imgLocationList.Count / BatchSize;
            inputTensorList = new List<Tensor>();
            labelTensorList = new List<Tensor>();

            Size = count;
        }

        private (Tensor image, Tensor Label) ReadImageAndLabelTensor(int idx, TorchSharp.torchvision.ITransform transform)
        {

            var transforms = new List<TorchSharp.torchvision.ITransform>();
            transforms.Add(TorchSharp.torchvision.transforms.Pad(new long[] { 0, 0 }, mode: PaddingModes.Reflect));
            transforms.Add(TorchSharp.torchvision.transforms.RandomHorizontalFlip());
            //transforms.Add(TorchSharp.torchvision.transforms.RandomResizedCrop(32));
            transforms.Add(TorchSharp.torchvision.transforms.Normalize(new double[] { 0.485, 0.456, 0.406 }, new double[] { 0.229, 0.224, 0.225 }));

            Random rnd = new Random(Guid.NewGuid().GetHashCode());
            var imgLocation = imgLocationList[idx];
            var label = labelList[idx];
            var mean = torch.tensor(new float[] { 0.485f, 0.456f, 0.406f }).reshape(1, 3, 1, 1);
            var var = torch.tensor(new float[] { 0.229f, 0.224f, 0.225f }).reshape(1, 3, 1, 1);
            var bmp = new Bitmap(imgLocation);
            BitmapData data = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppRgb);

            // copy as bytes
            int byteCount = data.Stride * data.Height;
            byte[] bytes = new byte[byteCount];
            System.Runtime.InteropServices.Marshal.Copy(data.Scan0, bytes, 0, byteCount);
            bmp.UnlockBits(data);

            using var midTensor0 = torch.tensor(bytes);
            //inputTensor = inputTensor.to(device.type, deviceIndex: device.index, copy: true);
            using var midTensor1 = midTensor0.@float();
            using var midTensor2 = midTensor1.reshape(1, bmp.Height, bmp.Width, 4);
            using var midTensor3 = midTensor2.transpose(0, 3);
            using var midTensor4 = midTensor3.reshape(4, bmp.Height, bmp.Width);
            var result1 = midTensor4.chunk(4, 0);

            //var result1 = inputTensor.@float().reshape(1, bmp.Height, bmp.Width, 4).transpose(0, 3).reshape(4, bmp.Height, bmp.Width).chunk(4, 0);
            List<Tensor> part = new List<Tensor>();
            if(reverse)
            {
                part.Add(result1[2]);
                part.Add(result1[1]);
                part.Add(result1[0]);
            }
            else
            {
                part.Add(result1[0]);
                part.Add(result1[1]);
                part.Add(result1[2]);
            }
            var inputTensor = torch.cat(part, 0);
            if(augment == true)
            {
                using var transMidTensor0 = inputTensor.reshape(1, 3, bmp.Height, bmp.Width);

                using var transMidTensor1 = transforms[0].forward(transMidTensor0);
                using var transMidTensor2 = transforms[1].forward(transMidTensor1);
                //using var transMidTensor3 = transforms[2].forward(transMidTensor2);
                using var oneTensor = torch.ones_like(transMidTensor2, dtype: transMidTensor2.dtype, device: transMidTensor2.device);
                using var T255Tensor = oneTensor * 255;
                using var transMidTensor4 = transMidTensor2 / T255Tensor;
                using var transMidTensor5 = transforms[2].forward(transMidTensor4);
                var transMidTensor6 = torch.nn.functional.upsample(transMidTensor5, new long[] { 224, 224 }, mode: UpsampleMode.Bilinear);
                return (transMidTensor6, torch.tensor(label, torch.int64, device: device));
            }
            else
            {
                using var transMidTensor0 = inputTensor.reshape(1, 3, bmp.Height, bmp.Width);
                using var transMidTensor4 = transMidTensor0 / 255;
                var transMidTensor5 = transforms[2].forward(transMidTensor4);
                var transMidTensor6 = torch.nn.functional.upsample(transMidTensor5, new long[] { 224, 224 }, mode: UpsampleMode.Bilinear);
                return (transMidTensor6, torch.tensor(label, torch.int64, device: device));
            }

            //var cropScale = rnd.NextDouble() * 0.92 + 0.08;
            //var upsampleRatio = rnd.NextDouble() * (1.333 - 0.75) + 0.75;
            //var newWidth = (int)Math.Sqrt((float)bmp.Width * bmp.Height * cropScale * upsampleRatio);
            //var newHeight = (int)Math.Sqrt((float)bmp.Width * bmp.Height * cropScale / upsampleRatio);

            //newWidth = Math.Min(Math.Max(0, newWidth), bmp.Width);
            //newHeight = Math.Min(Math.Max(0, newHeight), bmp.Height);

            //if (augment == true)
            //{
            //    inputTensor = inputTensor.split(new long[] { (bmp.Height - newHeight) / 2, newHeight, bmp.Height - newHeight - (bmp.Height - newHeight) / 2 }, 2)[1].split(new long[] { (bmp.Width - newWidth) / 2, newWidth, bmp.Width - newWidth - (bmp.Width - newWidth) / 2 }, 3)[1];
            //}
            //var properSizeTensor = torch.nn.functional.upsample(inputTensor, size: new long[] { 224, 224 }, mode: UpsampleMode.Bilinear) / 255;
            //inputTensor.Dispose();

            //if (rnd.Next() % 2 == 0 && augment == true)
            //{
            //    properSizeTensor = properSizeTensor.flip(new long[] { 2 });
            //}
            //using var midTensor5 = properSizeTensor.sub(mean);
            //var ret = midTensor5.div(var);
            //properSizeTensor.Dispose();

        }

        private void prepareInputTensorThread()
        {
            var innerTransform = TorchSharp.torchvision.transforms.RandomResizedCrop(224);
            //innerTransforms.Add(TorchSharp.torchvision.transforms.Normalize(new double[] { 0.485, 0.456, 0.406 }, new double[] { 0.229, 0.224, 0.225 }));
            while (true)
            {
                var batchIndexChosen = 0;
                lock (lockerNumber)
                {
                    batchIndexChosen = currentBatchIndex++;
                }
                if (batchIndexChosen >= batchCount)
                {
                    return;
                }
                var dataTensor = torch.zeros(new long[] { BatchSize, 3, 224, 224 }, device: device);
                var lablTensor = torch.zeros(new long[] { BatchSize }, torch.int64, device: device);

                for (var j = batchIndexChosen * BatchSize; j < batchIndexChosen * BatchSize + BatchSize; j++)
                {
                    var idx = indices[j];
                    //try
                    //{
                        var (imageTensor, labelTensor) = ReadImageAndLabelTensor(idx, innerTransform);
                        lablTensor[j - batchIndexChosen * BatchSize] = labelTensor;
                        dataTensor.index_put_(imageTensor, TensorIndex.Single(j - batchIndexChosen * BatchSize));
                    //}
                    //catch (Exception e)
                    //{
                    //    int lastImageIndex;
                    //    lock (lockerNumber)
                    //    {
                    //        lastImageIndex = currentLastIndex--;
                    //    }
                    //    Console.WriteLine($"Error occured in reading {idx}th image, message: {e.Message}, reading the {lastImageIndex}th image to replace it");
                    //    var (imageTensor, labelTensor) = ReadImageAndLabelTensor(lastImageIndex, innerTransform);
                    //    lablTensor[j - batchIndexChosen * BatchSize] = labelTensor;
                    //    dataTensor.index_put_(imageTensor, TensorIndex.Single(j - batchIndexChosen * BatchSize));
                    //}
                }

                //dataTensor = dataTensor.to(device);
                dataTensor = dataTensor.reshape(BatchSize, 3, 224, 224);
                var sleep_count = 0;
                while (true)
                {
                    lock (lockerDict)
                    {
                        if (inputTensorList.Count < 100)
                        {
                            break;
                        }
                        else
                        {
                            Thread.Sleep(100);
                            sleep_count++;
                        }
                    }
                }
                lock (lockerDict)
                {
                    inputTensorList.Add(dataTensor);
                    labelTensorList.Add(lablTensor);
                }
            }
        }

        public void reShuffle()
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());
            indices = !shuffle ?
                Enumerable.Range(0, Size).ToArray() :
                Enumerable.Range(0, Size).OrderBy(c => rnd.Next()).ToArray();
        }

        private object lockerNumber = new object();
        private object lockerDict = new object();

        private torch.Device device { get; set; }

        private bool shuffle { get; set; }

        private int[] indices { get; set; }
        public int batchCount { get; private set; }
        public int Size { get; set; }

        public int BatchSize { get; private set; }

        private int currentBatchIndex { get; set; }
        private int currentLastIndex { get; set; }

        private List<Tensor> inputTensorList { get; set; }
        private List<Tensor> labelTensorList { get; set; }

        private List<string> imgLocationList = new List<string>();
        private List<int> labelList = new List<int>();

        private bool reverse = true;
        private bool augment = true;

        public void Dispose()
        {
        }

        public void StartThreads(int threadCount = 8)
        {
            reShuffle();
            currentLastIndex = imgLocationList.Count - 1;
            currentBatchIndex = 0;
            prepareInputTensorThread();
            List<Thread> threads = new List<Thread>();
            for (int i = 0; i < threadCount; i++)
            {
                threads.Add(new Thread(new ThreadStart(prepareInputTensorThread)));
            }
            for (int i = 0; i < threadCount; i++)
            {
                threads[i].Start();
            }
        }

        public IEnumerable<(Tensor, Tensor)> Data()
        {
            for (int i = 0; i < batchCount; i++)
            {
                while (true)
                {
                    bool hasTensor = false;
                    lock (lockerDict)
                    {
                        hasTensor = inputTensorList.Count > 0 && labelTensorList.Count > 0;
                    }
                    if (hasTensor)
                    {
                        break;
                    }
                    else
                    {
                        Thread.Sleep(1000);
                    }
                }
                Tensor dataTensor, lablTensor;
                lock (lockerDict)
                {
                    dataTensor = (Tensor)inputTensorList.First();
                    lablTensor = (Tensor)labelTensorList.First();
                    inputTensorList.RemoveAt(0);
                    labelTensorList.RemoveAt(0);
                }
                yield return (dataTensor, lablTensor);
            }
        }
    }
}
