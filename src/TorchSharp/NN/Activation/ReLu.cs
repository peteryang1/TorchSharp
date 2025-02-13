// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a ReLU module.
        /// </summary>
        public class ReLU : torch.nn.Module
        {
            internal ReLU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_ReLU_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_ReLU_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(ReLU).Name;
            }
        }
    }
    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_ReLU_ctor(bool inplace, out IntPtr pBoxedModule);

            /// <summary>
            /// Rectified Linear Unit
            /// </summary>
            /// <param name="inPlace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            static public ReLU ReLU(bool inPlace = false)
            {
                var handle = THSNN_ReLU_ctor(inPlace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ReLU(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Rectified Linear Unit
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inPlace">Do the operation in-place. Default: False</param>
                /// <returns></returns>
                static public Tensor relu(Tensor x, bool inPlace = false)
                {
                    using (var m = nn.ReLU(inPlace)) {
                        return m.forward(x);
                    }
                }
            }
        }
    }
}
