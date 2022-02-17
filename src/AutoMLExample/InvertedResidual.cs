using System;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class InvertedResidual : Module
{
    private Module conv_pw;
    private Module bn1;
    private Module act1;
    private Module conv_dw;
    private Module bn2;
    private Module act2;
    private Module se;
    private Module conv_pwl;
    private Module bn3;

    public InvertedResidual(string name, dynamic arch) : base(name)
    {
        conv_pw = Loader.LoadArch(arch.conv_pw);
        bn1 = Loader.LoadArch(arch.bn1);
        act1 = Loader.LoadArch(arch.act1);
        conv_dw = Loader.LoadArch(arch.conv_dw);
        bn2 = Loader.LoadArch(arch.bn2);
        act2 = Loader.LoadArch(arch.act2);
        se = Loader.LoadArch(arch.se);
        conv_pwl = Loader.LoadArch(arch.conv_pwl);
        bn3 = Loader.LoadArch(arch.bn3);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        //using var residual = x;

        using var x1 = conv_pw.forward(x);
        using var x2 = bn1.forward(x1);
        using var x3 = act1.forward(x2);

        using var x4 = conv_dw.forward(x3);
        using var x5 = bn2.forward(x4);
        using var x6 = act2.forward(x5);

        using var x7 = se.forward(x6);

        using var x8 = conv_pwl.forward(x7);
        var x9 = bn3.forward(x8);

        // FIXME: residual is not implemented because it needs shape inference
        if (x9.shape[1] == x.shape[1] && x9.shape[2] == x.shape[2] && x9.shape[3] == x.shape[3])
        {
            x9 = x9 + x;
        }

        return x9;
    }
}
