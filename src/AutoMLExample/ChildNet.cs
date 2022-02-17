using System.Collections.Generic;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class ChildNet : Module
{
    private Module conv_stem;
    private Module bn1;
    private Module act1;
    private Module blocks1;
    private Module blocks2;
    private Module blocks3;
    private Module global_pool;
    private Module conv_head;
    private Module act2;
    private Module classifier;
    private Module flatten;

    public ChildNet(string name, dynamic arch) : base(name)
    {
        conv_stem = Loader.LoadArch(arch.conv_stem);
        bn1 = Loader.LoadArch(arch.bn1);

        act1 = Loader.LoadArch(arch.act1);
        blocks1 = Loader.LoadArch(arch.blocks1);
        blocks2 = Loader.LoadArch(arch.blocks2);
        blocks3 = Loader.LoadArch(arch.blocks3);
        global_pool = Loader.LoadArch(arch.global_pool);
        conv_head = Loader.LoadArch(arch.conv_head);
        act2 = Loader.LoadArch(arch.act2);
        classifier = Loader.LoadArch(arch.classifier);
        flatten = Flatten(1);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        RegisterComponents();
        // forward_features
        using var x1 = conv_stem.forward(x);
        using var x2 = bn1.forward(x1);
        using var x3 = act1.forward(x2);
        using var block1_x = blocks1.forward(x3);
        using var block2_x = blocks2.forward(block1_x);
        using var block3_x = blocks3.forward(block2_x);
        var ret = cat(new List<Tensor>() { functional.avg_pool2d(block1_x, new long[] { 2, 2 }, new long[] { 2, 2 }), block2_x, block3_x }, 1);
        return ret;
        //x = global_pool.forward(x);
        //x = conv_head.forward(x);
        //x = act2.forward(x);

        //x = flatten.forward(x);
        //// FIXME: dropout is a little tricky because we can't set it in constructor
        //return classifier.forward(x);
    }
}
