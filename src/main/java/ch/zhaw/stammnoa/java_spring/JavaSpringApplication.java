package ch.zhaw.stammnoa.java_spring;
import ai.djl.Model;
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


@SpringBootApplication
public class JavaSpringApplication {

    public JavaSpringApplication() throws TranslateException, IOException {
    }

    public static void main(String[] args) {
        SpringApplication.run(JavaSpringApplication.class, args);
    }


    ImageFolder dataset = initDataset("C:\\MDM\\ut-zap50k-images-square");
    RandomAccessDataset[] datasets = dataset.randomSplit(8, 2);

    RandomAccessDataset trainDataset = datasets[0];
    RandomAccessDataset validateDataset = datasets[1];



    public void data() throws TranslateException, IOException {
    }


    private static ImageFolder initDataset(String datasetRoot)
            throws IOException, TranslateException {

        ImageFolder dataset = ImageFolder.builder()

                // retrieve the data
                .setRepositoryPath(Paths.get(datasetRoot))
                .optMaxDepth(10)
                .addTransform(new Resize(32, 32))
                .addTransform(new ToTensor())
                // random sampling; don't process the data in order
                .setSampling(32, true)
                .build();
        dataset.prepare();
        return dataset;
    }


    // Modell initialisieren
    public static final String MODEL_NAME = "shoeclassifier";

    public static Model getModel() {
        // create new instance of an empty model
        Model model = Model.newInstance(MODEL_NAME);
        Block resNet50 =
                ResNetV1.builder() // construct the network
                        .setImageShape(new Shape(3, 32, 32))
                        .setNumLayers(50)
                        .setOutSize(10)
                        .build();
        // set the neural network to the model
        model.setBlock(resNet50);
        return model;
    }


    //Konfiguration Training
    Loss loss = Loss.softmaxCrossEntropyLoss();
    TrainingConfig config = setupTrainingConfig(loss);

    private static TrainingConfig setupTrainingConfig(Loss loss) {
        return new DefaultTrainingConfig(loss)
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());
    }

    train(trainDataset, validateDataset);
    //Training
    public void train(RandomAccessDataset trainDataset, RandomAccessDataset validateDataset) throws IOException, TranslateException {
        try ( Model model = getModel();
              Trainer trainer = model.newTrainer(config)) {
            trainer.setMetrics(new Metrics());
            Shape inputShape = new Shape(1, 3, 32, 32);
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



