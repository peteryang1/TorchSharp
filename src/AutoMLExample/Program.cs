using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace AutoMLExample
{
    public class Program
    {
        private static int _epochs = 50;
        private static int _epochs_scale = 2;

        private static int _trainBatchSize = 128;
        private static int _testBatchSize = 128;

        private static readonly int _logInterval = 50;
        internal static void TrainingLoop(string dataset, Device device, Module backbone, Module clsHead, DataReader train, DataReader test)
        {
            backbone = backbone.to(device);
            clsHead = clsHead.to(device);

            var initLR = 0.1;
            var optimizer = torch.optim.SGD(clsHead.parameters(), learningRate: initLR, weight_decay: 5e-4);
            //var scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, new List<int> { 15, 30, 40 }, 0.1);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            for (var epoch = 1; epoch <= _epochs * _epochs_scale; epoch++)
            {
                if(epoch <= 15 * _epochs_scale)
                {
                    optimizer.LearningRate = initLR;
                }
                else if (15 * _epochs_scale < epoch && epoch <= 30 * _epochs_scale)
                {
                    optimizer.LearningRate = initLR * 0.2;
                }
                else if (30 * _epochs_scale < epoch && epoch <= 40 * _epochs_scale)
                {
                    optimizer.LearningRate = initLR * 0.04;
                }
                else if (40 * _epochs_scale < epoch && epoch <= 50 * _epochs_scale)
                {
                    optimizer.LearningRate = initLR * 0.008;
                }
                Train(backbone, clsHead, optimizer, cross_entropy_loss(reduction: Reduction.Mean), device, train, epoch, train.BatchSize, train.Size);

                clsHead.eval();
                Console.WriteLine("Saving model to '{0}'", dataset + epoch + ".model.bin");
                clsHead.save(dataset + epoch + ".model.bin");
                Test(backbone, clsHead, cross_entropy_loss(reduction: torch.nn.Reduction.Sum), device, test, test.Size);
                //if (File.Exists(dataset + "-" + epoch + ".model.bin"))
                //{
                //    Console.WriteLine("Loading model from '{0}'", dataset + ".model.bin");
                //    Console.WriteLine(dataset + "-" + epoch + ".model.bin");
                //    clsHead = clsHead.load(dataset + epoch + ".model.bin");
                //}
                //else
                //{
                //}

                Console.WriteLine($"End-of-epoch memory use: {GC.GetTotalMemory(false)}");
                //scheduler.step();
            }
            Test(backbone, clsHead, cross_entropy_loss(reduction: torch.nn.Reduction.Sum), device, test, test.Size);
            sw.Stop();
            Console.WriteLine($"Elapsed time: {sw.Elapsed.TotalSeconds:F1} s.");
        }

        private static void Train(
            Module backbone,
            Module clsHead,
            torch.optim.Optimizer optimizer,
            Loss loss,
            Device device,
            DataReader dataLoader,
            int epoch,
            long batchSize,
            long size)
        {
            Stopwatch dataSW = new Stopwatch();
            Stopwatch modelSW = new Stopwatch();
            Stopwatch allSw = new Stopwatch();

            backbone.eval();
            clsHead.train();

            int batchId = 1;

            Console.WriteLine($"Epoch: {epoch}...");
            dataSW.Start();
            allSw.Start();

            dataLoader.StartThreads(6);
            int correctCount = 0;
            long allCount = 0;
            foreach (var (data, target) in dataLoader.Data())
            {
                var gpuData = data.to(device);
                optimizer.zero_grad();
                dataSW.Stop();
                modelSW.Start();
                //Tensor prediction;

                var prediction = clsHead.forward(backbone.forward(gpuData));
                //while (true)
                //{
                //    try
                //    {
                //        break;
                //    }
                //    catch (System.Runtime.InteropServices.ExternalException)
                //    {
                //        Console.WriteLine("error detected in inference.");
                //    }
                //}
                modelSW.Stop();
                var output = loss(prediction, target);

                output.backward();

                optimizer.step();

                if (batchId % _logInterval == 0)
                {
                    correctCount += (prediction.argmax(1).eq(target).sum()).ToInt32();
                    allCount += batchSize;
                    var acc = correctCount / (float)allCount;
                    allSw.Stop();
                    Console.WriteLine($"\rTrain: epoch {epoch}, batch_iteration {batchId} [{batchId * batchSize} / {size}], LearningRate: {((TorchSharp.Modules.SGDOptimizer)optimizer).LearningRate}, Loss: {output.ToSingle():F4}, Acc: {acc}, dataTime: {dataSW.Elapsed.TotalSeconds} seconds, inference Time: {modelSW.Elapsed.TotalSeconds} seconds, all time: {allSw.Elapsed.TotalSeconds} seconds.");
                    dataSW.Reset();
                    modelSW.Reset();
                    allSw.Restart();
                    correctCount = 0;
                    allCount = 0;
                }

                batchId++;
                gpuData.Dispose();
                data.Dispose();
                target.Dispose();

                GC.Collect();
                dataSW.Start();
            }
        }

        private static void Test(
            Module backbone,
            Module clsHead,
            Loss loss,
            Device device,
            DataReader dataLoader,
            long size)
        {
            clsHead.eval();
            backbone.eval();

            double testLoss = 0;
            int correct = 0;

            dataLoader.StartThreads();
            foreach (var (data, target) in dataLoader.Data())
            {
                var prediction = clsHead.forward(backbone.forward(data));
                var output = loss(prediction, target);
                testLoss += output.ToSingle();

                var pred = prediction.argmax(1);
                correct += pred.eq(target).sum().ToInt32();

                pred.Dispose();

                GC.Collect();
            }

            Console.WriteLine($"Size: {size}, Total: {size}");

            Console.WriteLine($"\rTest set: Average loss {(testLoss / size):F4} | Accuracy {((double)correct / size):P2}");
        }

        public static void Main(string[] args)
        {
            var device = torch.cuda.is_available() ? new torch.Device("CUDA", 0) : torch.CPU;
            //var device = torch.CPU;
            //var device = torch.CPU;
            var convTmp = torch.nn.Conv2d(16, 16, 3);
            var model1 = ResNet.ResNet34(1000, device);
            var model = Loader.LoadArchAndWeights(@"F:\workspace");
            model.eval();
            nn.Module classifierHead = new classificationHead(100);
            classifierHead.eval();

            //var resModel = ResNet.ResNet18(100).to(device);

            model = model.to(device);
            classifierHead = classifierHead.to(device);
            //var inputTensor = torch.ones(new long[] { 8, 3, 224, 224 }).to(device);
            ////var output = classifierHead.forward(model.forward(inputTensor));
            //var output = resModel.forward(inputTensor);

            //DateTime beforDT = System.DateTime.Now;

            //for(int i=0; i<50;i++)
            //{
            //    output = resModel.forward(inputTensor);
            //    //output = classifierHead.forward(model.forward(inputTensor));
            //}

            //DateTime afterDT = System.DateTime.Now;
            //TimeSpan ts = afterDT.Subtract(beforDT);
            //Console.WriteLine("DateTime总共花费{0}ms.", ts.TotalMilliseconds);
            //return;

            //var transforms = new List<TorchSharp.torchvision.ITransform>();
            //transforms.Add(TorchSharp.torchvision.transforms.RandomResizedCrop(224));
            //transforms.Add(TorchSharp.torchvision.transforms.RandomHorizontalFlip());
            //transforms.Add(TorchSharp.torchvision.transforms.Normalize(new double[] { 0.485, 0.456, 0.406 }, new double[] { 0.229, 0.224, 0.225 }));

            using (DataReader train = new DataReader(@"F:\workspace\archive", @"seg_train\seg_train", _trainBatchSize, device: device, shuffle: true, reverseGBR: false, augmentation:true),
                                test = new DataReader(@"F:\workspace\archive", @"seg_test\seg_test", _testBatchSize, device: device, reverseGBR: false, augmentation:false))
            {

                TrainingLoop(@"F:\workspace\cifar_models\", device, model, classifierHead, train, test);
            }
        }
    }
}
