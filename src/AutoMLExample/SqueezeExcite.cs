using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class SqueezeExcite : Module
{
    private Module avg_pool;
    private Module conv_reduce;
    private Module act1;
    private Module conv_expand;

    public SqueezeExcite(string name, dynamic arch) : base(name)
    {
        avg_pool = Loader.LoadArch(arch.avg_pool);
        conv_reduce = Loader.LoadArch(arch.conv_reduce);
        act1 = Loader.LoadArch(arch.act1);
        conv_expand = Loader.LoadArch(arch.conv_expand);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var xe1 = avg_pool.forward(x);
        using var xe2 = conv_reduce.forward(xe1);
        using var xe3 = act1.forward(xe2);
        using var xe4 = conv_expand.forward(xe3);

        // FIXME: asserting gate_fn is sigmoid
        var x5 = x * xe4.hardsigmoid();
        return x5;
    }
}
