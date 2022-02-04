package p;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class cosas {
	
	static long startTime = System.nanoTime();

    public static void main(String[] args) throws Exception {
    		
    	    DataSource source = new DataSource(args[0]);
            Instances data = source.getDataSet();
            
            data.setClassIndex(data.numAttributes() -1);
            
            NaiveBayes model = new NaiveBayes(); //Construye el modelo.
	        Evaluation eval = new Evaluation(data);
	        eval.crossValidateModel(model, data, 5, new Random(1));
	        eval.toMatrixString();
	        
	        System.out.println("-------------------------------------------------------------");
	        System.out.println("Datu sorta honetan " + data.numInstances() + " instantzia daude.");
	        System.out.println("Datu sorta honetan " + data.numAttributes() + " atributu daude.");
	        System.out.println("Datu sorta honetan, lehenengo atributuak  " + data.numDistinctValues(0) + " balio desberdin hartu ditzake.");
	        System.out.println("Datu sorta honetan, azken-aurreko atributuak  "  + data.attributeStats(data.numAttributes() - 2).missingCount + " missing value ditu.");
	        System.out.println("-------------------------------------------------------------");
	        
	        File f = new File(args[1]);
	        try {
	            f.createNewFile();
	            FileWriter myWriter = new FileWriter(args[1]);
	            long endTime   = System.nanoTime();
	            long totalTime = endTime - startTime;
	            myWriter.write("Execution time: "+totalTime/1000 +" miliseconds.");
	            myWriter.write("\n");
	            myWriter.write("Created file directory: " +args[1]+"\n");
	            myWriter.write("\n");
	            myWriter.write(eval.toMatrixString());
	            myWriter.flush();
	            myWriter.close();
	        } catch (Exception e) {
	            e.printStackTrace();
	        }     
    }
	        
}
