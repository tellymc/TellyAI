package me.tellymc.layers;

import me.tellymc.GPUMath;
import me.tellymc.objects.Tensor;

import java.util.Random;

public class LinearLayer {

    private final int inputSize;
    private final int outputSize;
    private final Tensor h1;
    private final Tensor weights;
    private final Tensor biases;
    private Tensor lastInput;
    private Tensor lastOutput;

    public LinearLayer(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        this.h1 = new Tensor(1, this.outputSize);
        this.weights = new Tensor(this.inputSize, this.outputSize);
        this.biases = new Tensor(1, this.outputSize);

        float[] h1List = new float[this.outputSize];
        float[] weightsList = new float[this.inputSize * this.outputSize];
        float[] biasesList = new float[this.outputSize];

        Random random = new Random();

        for (int i = 0; i < this.outputSize; i++) {
            h1List[i] = random.nextFloat();
            biasesList[i] = random.nextFloat();
        }

        for (int i = 0; i < weightsList.length; i++) {
            weightsList[i] = random.nextFloat();
        }

        this.h1.upload(h1List);
        this.weights.upload(weightsList);
        this.biases.upload(biasesList);
    }

    public void forward(Tensor input, Tensor output) {

        if (input.getRows() != 1 || input.getColumns() != this.inputSize) {
            System.out.println("Input has the incorrect dimensions. Correct: 1x" + this.inputSize);
            return;
        }

        if (output.getRows() != 1 || output.getColumns() != this.outputSize) {
            System.out.println("Output has the incorrect dimensions. Correct: 1x" + this.outputSize);
            return;
        }

        GPUMath.multiply(input, this.weights, this.h1);
        GPUMath.add(this.h1, this.biases, output);

        this.lastInput = input;
        this.lastOutput = output;
    }

    public void close() {
        h1.close();
        weights.close();
        biases.close();
        lastInput.close();
        lastOutput.close();
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }
}