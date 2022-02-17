using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class SelectAdaptivePool2d : Module
{
    private Module pool;

    public SelectAdaptivePool2d(string name, dynamic arch) : base(name)
    {
        pool = Loader.LoadArch(arch.pool);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // FIXME: asserting flatten = False
        var x1 = pool.forward(x);
        return x1;
    }
}
