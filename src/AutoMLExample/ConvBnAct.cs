using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class ConvBnAct : Module
{
    private Module conv;
    private Module bn1;
    private Module act1;

    public ConvBnAct(string name, dynamic arch) : base(name)
    {
        conv = Loader.LoadArch(arch.conv);
        bn1 = Loader.LoadArch(arch.bn1);
        act1 = Loader.LoadArch(arch.act1);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var x1 = conv.forward(x);
        using var x2 = bn1.forward(x1);
        var x3 = act1.forward(x2);
        return x3;
    }
}
