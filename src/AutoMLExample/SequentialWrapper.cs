using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class SequentialWrapper : Module
{
    private Module sequential;
    private Module[] children;

    public SequentialWrapper(string name, dynamic arch) : base(name + "_wrapper")
    {
        var modules = new List<Module>();
        foreach (var module_arch in (JArray)arch)
        {
            modules.Add(Loader.LoadArch(module_arch));
        }

        sequential = Sequential(modules);
        RegisterComponents();

        children = modules.ToArray();
    }

    public override Tensor forward(Tensor x)
    {
        var x1 = sequential.forward(x);
        return x1;
    }

    public Module this[int index]
    {
        get { return children[index]; }
    }
}
