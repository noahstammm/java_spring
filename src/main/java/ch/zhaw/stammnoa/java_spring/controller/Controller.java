package ch.zhaw.stammnoa.java_spring.controller;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.ModelAndView;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Base64;

@RestController
public class Controller {
    private final Logger logger = org.slf4j.LoggerFactory.getLogger(Controller.class);


    @GetMapping("/")
    public ModelAndView index() {
        logger.info("GET / called");
        return new ModelAndView("index");
    }

    @PostMapping("/upload")
    public ModelAndView handleFileUpload(@RequestParam("file") MultipartFile file) throws IOException, TranslateException {
        logger.info("POST /upload called");

        ModelAndView modelAndView = new ModelAndView("prediction");


        return modelAndView;
    }

    private String getBase64Image(Image image) throws IOException {
        BufferedImage bufferedImage = (BufferedImage) image.getWrappedImage();
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        ImageIO.write(bufferedImage, "png", outputStream);
        return Base64.getEncoder().encodeToString(outputStream.toByteArray());
    }
}
