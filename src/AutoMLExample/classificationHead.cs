using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class classificationHead : Module
{
    private Module conv1;
    private Module globalPool;
    private Module convHead;
    private Module act;
    private Module classifier;
    private Module flatten;

    public classificationHead(int numClasses = 1000,
            int numFeatures= 1280) : base("classificationHead")
    {
        conv1 = Conv2d(608, numFeatures, 1);
        globalPool = AvgPool2d(new long[] { 7, 7 });
        convHead = Conv2d(numFeatures, numFeatures, 1);
        act = ReLU(inPlace: true);
        flatten = Flatten(1);
        classifier = Linear(numFeatures, numClasses);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var x1 = conv1.forward(x);
        using var x2 = globalPool.forward(x1);
        using var x3 = convHead.forward(x2);
        using var x4 = act.forward(x3);
        using var x5 = flatten.forward(x4);
        var x6 = classifier.forward(x5);

        return x6;
    }
}
