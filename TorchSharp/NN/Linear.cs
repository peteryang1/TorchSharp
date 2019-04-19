﻿using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class Linear : ProvidedModule
    {
        public Linear(IntPtr handle) : base(handle)
        {
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_linearModule(long input_size, long output_size, bool with_bias);

        public Linear(long inputSize, long outputSize, bool hasBias = false) : base()
        {
            handle = new HType(THSNN_linearModule(inputSize, outputSize, hasBias), true);
        }

        [DllImport("libTorchSharp")]
        extern static bool THSNN_linear_with_bias(Module.HType module);

        public bool WithBias
        {
            get { return THSNN_linear_with_bias(handle); }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_linear_get_bias(Module.HType module);

        [DllImport("libTorchSharp")]
        extern static void THSNN_linear_set_bias(Module.HType module, IntPtr tensor);

        public ITorchTensor<float> Bias
        {
            get
            {
                var bias = THSNN_linear_get_bias(handle);
                if (bias == IntPtr.Zero)
                {
                    throw new ArgumentNullException("Linear module without bias term.");
                }
                return new FloatTensor(bias);
            }
            set { THSNN_linear_set_bias(handle, value.Handle); }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_linear_get_weight(Module.HType module);

        [DllImport("libTorchSharp")]
        extern static void THSNN_linear_set_weight(Module.HType module, IntPtr tensor);

        public ITorchTensor<float> Weight
        {
            get
            {
                return new FloatTensor(THSNN_linear_get_weight(handle));
            }
            set { THSNN_linear_set_weight(handle, value.Handle); }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_linearModuleApply(Module.HType module, IntPtr tensor);

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            return new FloatTensor(THSNN_linearModuleApply(handle, tensor.Handle));
        }
    }
}
