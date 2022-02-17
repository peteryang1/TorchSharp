using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class Swish : Module
{
    public Swish(string name, dynamic arch) : base(name)
    {
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var x1 = x.sigmoid();
        var x2 = x.mul(x1);
        return x2;
        //return x.mul(x.sigmoid());
    }
}
