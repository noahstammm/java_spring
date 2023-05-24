package ch.zhaw.stammnoa.java_spring;
import ch.zhaw.stammnoa.java_spring.controller.Model;
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

    public static void main(String[] args) throws TranslateException, IOException {
        SpringApplication.run(JavaSpringApplication.class, args);

        // Instanz der Model-Klasse erstellen
        Model model = new Model();
        final String MODEL_NAME = "shoeclassifier";

        ImageFolder dataset = model.initDataset("C:/MDM/ut-zap50k-images-square");
        RandomAccessDataset[] datasets = dataset.randomSplit(8, 2);

        RandomAccessDataset trainingSet = datasets[0];
        RandomAccessDataset testSet = datasets[1];

        model.getModel(MODEL_NAME);

        //Konfiguration Training
         Loss loss = Loss.softmaxCrossEntropyLoss();
         TrainingConfig config = model.setupTrainingConfig(loss);

        // Methode aufrufen
        model.train(trainingSet, testSet, config, MODEL_NAME);



    }

}




