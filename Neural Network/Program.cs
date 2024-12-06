using System;
using System.Linq;
using System.Collections.Generic;

//Using my own diferentiation engine
using SharpGrad;
using SharpGrad.DifEngine; 

public class Layer
{
    public Value<float>[,] Weights; 
    public Value<float>[] Biases;
    private int inputSize;
    private int outputSize;
    private Random rnd;

    public Layer(int inputSize, int outputSize, Random rnd)
    {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.rnd = rnd;

        Weights = new Value<float>[inputSize, outputSize];
        Biases = new Value<float>[outputSize];

        for(int i=0;i<inputSize;i++)
        {
            for (int j=0;j<outputSize;j++)
            {
                Weights[i,j]=new Value<float>((float)(rnd.NextDouble()*2.0-1.0), $"W_{i}_{j}");
            }
        }

        for (int j=0;j<outputSize;j++)
        {
            Biases[j]=new Value<float>((float)(rnd.NextDouble()*2.0-1.0),$"b_{j}");
        }
    }

    public Value<float>[] Forward(Value<float>[] input)
    {
        Value<float>[] output=new Value<float>[outputSize];
        for (int j=0; j<outputSize; j++)
        {
            Value<float> sum=new Value<float>(0.0f,"sum");
            for (int i=0;i<inputSize;i++)
            {
                sum=sum+(input[i]*Weights[i,j]);
            }
            sum=sum+Biases[j];
            output[j]=sum;
        }
        return output;
    }
}

public class NeuralNetwork
{
    private List<Layer>layers;
    private Func<Value<float>,Value<float>>activationHidden; 
    private Func<Value<float>,Value<float>>activationOutput;

    public NeuralNetwork(int inputSize,int hiddenSize,int outputSize,Random rnd)
    {
        layers = new List<Layer>();
        layers.Add(new Layer(inputSize,hiddenSize,rnd));
        layers.Add(new Layer(hiddenSize,outputSize,rnd));

        activationHidden=(x)=>x.Tanh();
        activationOutput=(x)=>x; 
    }

    public Value<float>[] Forward(Value<float>[] input)
    {
        var h=layers[0].Forward(input).Select(v=>activationHidden(v)).ToArray();
        var o=layers[1].Forward(h).Select(v=>activationOutput(v)).ToArray();
        return o;
    }

    public List<Value<float>> GetParameters()
    {
        var parameters=new List<Value<float>>();
        foreach(var layer in layers)
        {
            foreach(var w in layer.Weights)
                parameters.Add(w);
            foreach(var b in layer.Biases)
                parameters.Add(b);
        }
        return parameters;
    }
}

class Program
{
    // y = sin(x) + 0.5*cos(2x)
    static float Func(float x)
    {
        return (float)(Math.Sin(x)+0.5*Math.Cos(2*x));
    }

    static void Main(string[] args)
    {
        Random rnd=new Random(42);

        NeuralNetwork net=new NeuralNetwork(1,20,1,rnd);

        int dataSize=100;
        float[] xs=new float[dataSize];
        float[] ys=new float[dataSize];
        for (int i=0;i<dataSize;i++)
        {
            float x=(float)(i*0.1);
            xs[i]=x;
            ys[i]=Func(x);
        }

        int epochs=10000;
        float learningRate=0.001f;

        List<Value<float>>parameters=net.GetParameters();

        for (int epoch=0;epoch<epochs;epoch++)
        {
            foreach(var p in parameters)
            {
                p.Grad=0.0f;
            }

            Value<float> totalLoss=new Value<float>(0.0f, "loss");

            for (int i = 0; i < dataSize; i++)
            {
                Value<float> xVal=new Value<float>(xs[i], $"x_{i}");
                var pred = net.Forward(new Value<float>[] { xVal })[0];
                Value<float> yVal=new Value<float>(ys[i], $"y_{i}");
                var diff=pred-yVal;
                var sq=diff*diff;
                totalLoss=totalLoss+sq;
            }

            totalLoss = totalLoss/new Value<float>(dataSize,"N");

            totalLoss.Grad = 1.0f;
            totalLoss.Backpropagate();

            foreach (var p in parameters)
            {
                p.Data=p.Data-learningRate*p.Grad;
            }

            if(epoch%500==0)
            {
                Console.WriteLine($"Epoch {epoch}, Loss: {totalLoss.Data}");
            }
        }

        Console.WriteLine("=== Before Training ===");
        for (float testX=0;testX<=10;testX+=2f)
        {
            Value<float>xTest = new Value<float>(testX,"xTest");
            var pred = net.Forward(new Value<float>[]{xTest})[0].Data;
            float actual = Func(testX);
            Console.WriteLine($"x={testX}, pred={pred}, actual={actual}");
        }
    }
}
