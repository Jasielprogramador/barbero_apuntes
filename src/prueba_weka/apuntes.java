package p;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.unsupervised.instance.Randomize;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;
import java.io.FileWriter;
import java.util.Random;

	public class main {
		
	static long startTime = System.nanoTime();

	    public static void main(String[] args) throws Exception {
	    		
	    	    DataSource source = new DataSource(args[0]);
	            Instances data = source.getDataSet();
	            
	            //###################################################
		    	
		        //Hay que ver donde esta la clase en el .arff, si esta al final:
	            data.setClassIndex(data.numAttributes() -1);
	            
	            //Si no esta al final y por lo contrario esta al principio:
	            data.setClassIndex(0);
	            
	            //###################################################

	            //---ESTO ES CON HOLD OUT---
	            trainHoldOut(data);
	            
	            //---ESTO CON CROSS VALIDATION---
	            trainCrossValidation(data);
	            
	            //---ESTO CON EBALUAZIO ZINTZOA---
	            trainEbaluazioZintzoa(data);
	            
	            //###################################################

	          
	    }
  
	    //ESTO PERTENECE A HOLD OUT 
	    private static Evaluation trainHoldOut(Instances data) throws Exception {
	          
	        Randomize filter = new Randomize();
            filter.setInputFormat(data);
            Instances RandomData = Filter.useFilter(data,filter);
            
            //Split the data.
            RemovePercentage filterRemove = new RemovePercentage();
            filterRemove.setInputFormat(RandomData); //Preparas el filtro.
            filterRemove.setPercentage(30); //Ajuste 1.
            
            Instances train = Filter.useFilter(RandomData,filterRemove);
            System.out.println("Train tiene estas instancias "+ train.numInstances());

            filterRemove = new RemovePercentage(); //Creas nueva instancia, para poder cambiar parametros.
            filterRemove.setInputFormat(RandomData);
            filterRemove.setPercentage(30);
            filterRemove.setInvertSelection(true);
            Instances test = Filter.useFilter(RandomData,filterRemove);
            System.out.println("Test tiene estas instancias "+ test.numInstances());
            datuakInprimatu(RandomData);
            // fitxategiaSortu(train(RandomData), args[1]);
            
	        test.setClassIndex(test.numAttributes() - 1);
	        
            
	        //###################################################
            
	        //CON NAIVE BAYES
	        NaiveBayes model1 = new NaiveBayes(); //Construye el modelo.
	        model1.buildClassifier(data); //Esto en crossValidation sobra.
	        Evaluation eval1 = new Evaluation(data);
	        eval1.evaluateModel(model1,test);
	        eval1.toMatrixString();
	        
	        //CON ZeroR
	        ZeroR model2 = new ZeroR(); //Construye el modelo.
	        model2.buildClassifier(data); //Esto en crossValidation sobra.
	        Evaluation eval2 = new Evaluation(data);
	        eval2.evaluateModel(model2,test);
	        eval2.toMatrixString();
	        
	        
	        //CON OneR
	        OneR model3 = new OneR();
	        model2.buildClassifier(data); //Esto en crossValidation sobra.
	        Evaluation eval3 = new Evaluation(data);
	        eval3.evaluateModel(model3,test);
	        eval3.toMatrixString();
	        
	        //###################################################

	        System.out.println("Accuracy "+eval1.pctCorrect());
	        System.out.println("F-measure "+eval1.weightedFMeasure());
	        System.out.println(eval1.toSummaryString("\nResults\n======\n", false));
	        System.out.println("Estimated Accuracy: " + Double.toString(eval1.pctCorrect()));
	        System.out.println("Estimated Accuracy: " + eval1.toMatrixString("num"));

	        return eval1;
	    }
	    
	    //ESTO PERTENECE A CROSS VALIDATION
	    private static Evaluation trainCrossValidation(Instances data) throws Exception {

	        //###################################################
            
	        //CON NAIVE BAYES
	        NaiveBayes model1 = new NaiveBayes(); //Construye el modelo.
	        Evaluation eval1 = new Evaluation(data);
	        eval1.crossValidateModel(model1, data, 5, new Random(1));
	        eval1.toMatrixString();
	        
	        //CON ZeroR
	        ZeroR model2 = new ZeroR(); //Construye el modelo.
	        Evaluation eval2 = new Evaluation(data);
	        eval2.crossValidateModel(model1, data, 5, new Random(1));
	        eval2.toMatrixString();
	        
	        
	        //CON OneR
	        OneR model3 = new OneR();
	        Evaluation eval3 = new Evaluation(data);
	        eval3.crossValidateModel(model1, data, 5, new Random(1));
	        eval3.toMatrixString();
	        
	        //###################################################

	        
	        System.out.println("Accuracy "+eval1.pctCorrect());
	        System.out.println("F-measure "+eval1.weightedFMeasure());
	        System.out.println(eval1.toSummaryString("\nResults\n======\n", false));
	        System.out.println("Estimated Accuracy: " + Double.toString(eval1.pctCorrect()));
	        System.out.println("Estimated Accuracy: " + eval1.toMatrixString("num"));
	        return eval1;
	    }
	    
	    private static Evaluation trainEbaluazioZintzoa(Instances data) throws Exception {
	    	 	
	        //###################################################
            
	        //CON NAIVE BAYES
	        NaiveBayes model1 = new NaiveBayes(); //Construye el modelo.
	        model1.buildClassifier(data); //Esto en crossValidation sobra.
	        Evaluation eval1 = new Evaluation(data);
	        eval1.evaluateModel(model1,data);
	        eval1.toMatrixString();
	        
	        //CON ZeroR
	        ZeroR model2 = new ZeroR(); //Construye el modelo.
	        model2.buildClassifier(data); //Esto en crossValidation sobra.
	        Evaluation eval2 = new Evaluation(data);
	        eval2.evaluateModel(model2,data);
	        eval2.toMatrixString();
	        
	        
	        //CON OneR
	        OneR model3 = new OneR();
	        model2.buildClassifier(data); //Esto en crossValidation sobra.
	        Evaluation eval3 = new Evaluation(data);
	        eval3.evaluateModel(model3,data);
	        eval3.toMatrixString();
	        
	        //###################################################
	        
	        System.out.println("Accuracy "+eval1.pctCorrect());
	        System.out.println("F-measure "+eval1.weightedFMeasure());
	        System.out.println(eval1.toSummaryString("\nResults\n======\n", false));
	        System.out.println("Estimated Accuracy: " + Double.toString(eval1.pctCorrect()));
	        System.out.println("Estimated Accuracy: " + eval1.toMatrixString("num"));
	        return eval1;
	    }
	    
	    //PAJA
	    
	    private static void datuakInprimatu(Instances data) {
	        System.out.println("-------------------------------------------------------------");
	        System.out.println("Datu sorta honetan " + data.numInstances() + " instantzia daude.");
	        System.out.println("Datu sorta honetan " + data.numAttributes() + " atributu daude.");
	        System.out.println("Datu sorta honetan, lehenengo atributuak  " + data.numDistinctValues(0) + " balio desberdin hartu ditzake.");
	        System.out.println("Datu sorta honetan, azken-aurreko atributuak  "  + data.attributeStats(data.numAttributes() - 2).missingCount + " missing value ditu.");
	        System.out.println("-------------------------------------------------------------");
	    }

	    private static void fitxategiaSortu(Evaluation eval, String directory) {
	        File f = new File(directory);
	        try {
	            f.createNewFile();
	            FileWriter myWriter = new FileWriter(directory);
	            long endTime   = System.nanoTime();
	            long totalTime = endTime - startTime;
	            myWriter.write("Execution time: "+totalTime/1000 +" miliseconds.");
	            myWriter.write("\n");
	            myWriter.write("Created file directory: " +directory+"\n");
	            myWriter.write("\n");
	            myWriter.write(eval.toMatrixString());
	            myWriter.flush();
	            myWriter.close();
	        } catch (Exception e) {
	            e.printStackTrace();
	        }

	    }
	}
