using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class DepthwiseSeparableConv : Module
{
    private Module conv_dw;
    private Module bn1;
    private Module act1;
    private Module se;
    private Module conv_pw;
    private Module bn2;
    private Module act2;

    public DepthwiseSeparableConv(string name, dynamic arch) : base(name)
    {
        conv_dw = Loader.LoadArch(arch.conv_dw);
        bn1 = Loader.LoadArch(arch.bn1);
        act1 = Loader.LoadArch(arch.act1);
        se = Loader.LoadArch(arch.se);
        conv_pw = Loader.LoadArch(arch.conv_pw);
        bn2 = Loader.LoadArch(arch.bn2);
        act2 = Loader.LoadArch(arch.act2);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        //using var residual = x;

        using var x1 = conv_dw.forward(x);
        using var x2 = bn1.forward(x1);
        using var x3 = act1.forward(x2);

        using var x4 = se.forward(x3);

        using var x5 = conv_pw.forward(x4);
        using var x6 = bn2.forward(x5);
        using var x7 = act2.forward(x6);

        var x8 = x7 + x;

        // FIXME: residual is not implemented because it needs shape inference

        return x8;
    }
}
