package ch.zhaw.stammnoa.java_spring.controller;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.*;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import java.io.IOException;
import java.nio.file.Paths;

public class Model {
    public static ImageFolder initDataset(String datasetRoot)
            throws IOException, TranslateException {

        ImageFolder dataset = ImageFolder.builder()

                // retrieve the data
                .setRepositoryPath(Paths.get(datasetRoot))
                .optMaxDepth(10)
                .addTransform(new Resize(224, 224))
                .addTransform(new ToTensor())
                // random sampling; don't process the data in order
                .setSampling(32, true)
                .build();
        dataset.prepare();
        return dataset;
    }


    // Modell initialisieren
    public static final String MODEL_NAME = "shoeclassifier";

    public static ai.djl.Model getModel(String name) {
        // create new instance of an empty model
        ai.djl.Model model = ai.djl.Model.newInstance(MODEL_NAME);
        Block resNet50 =
                ResNetV1.builder() // construct the network
                        .setImageShape(new Shape(3, 224, 224))
                        .setNumLayers(50)
                        .setOutSize(10)
                        .build();
        // set the neural network to the model
        model.setBlock(resNet50);
        return model;
    }


    public static TrainingConfig setupTrainingConfig(Loss loss) {
        return new DefaultTrainingConfig(loss)
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());
    }


    //Training


    public static void train(RandomAccessDataset trainDataset, RandomAccessDataset validateDataset, TrainingConfig config,String name) throws IOException, TranslateException {
        try (ai.djl.Model model = getModel(name);
             Trainer trainer = model.newTrainer(config)) {
            trainer.setMetrics(new Metrics());
            Shape inputShape = new Shape(1, 3, 224, 224);
            trainer.initialize(inputShape);
            EasyTrain.fit(trainer, 1, trainDataset, validateDataset);
            TrainingResult result = trainer.getTrainingResult();
            model.setProperty("Epoch", String.valueOf(1));
            model.setProperty("Accuracy", String.format("%.5f", result.getValidateEvaluation("Accuracy")));
            model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
            model.save(Paths.get(Paths.get("src", "main", "model").toString()), MODEL_NAME);

        }
    }


}



