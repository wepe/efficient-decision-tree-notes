package main;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Main {
    public static void main(String[] args){
        //arguments　parser
        String file_training = args[0];
        String file_validation = args[1];
        String file_testing = args[2];
        int early_stopping_round = Integer.parseInt(args[3]);
        boolean maximize = args[4].equals("true");
        String eval_metric = args[5];
        String loss = args[6];
        double eta = Double.parseDouble(args[7]);
        int num_boost_round = Integer.parseInt(args[8]);
        int max_depth = Integer.parseInt(args[9]);
        double scale_pos_weight = Double.parseDouble(args[10]);
        double rowsample = Double.parseDouble(args[11]);
        double colample = Double.parseDouble(args[12]);
        double min_child_weight = Double.parseDouble(args[13]);
        int min_sample_split = Integer.parseInt(args[14]);
        double lambda = Double.parseDouble(args[15]);
        double gamma = Double.parseDouble(args[16]);
        int num_thread = Integer.parseInt(args[17]);

        GBM tgb = new GBM();
        tgb.fit(file_training,
                file_validation,
                early_stopping_round,
                maximize,
                eval_metric,
                loss,
                eta,
                num_boost_round,
                max_depth,
                scale_pos_weight,
                rowsample,
                colample,
                min_child_weight,
                min_sample_split,
                lambda,
                gamma,
                num_thread);

        double[] preds = tgb.predict(new TestData(file_testing).origin_feature);

        String[] strs = new String[preds.length];
        for(int i=0;i<strs.length;i++){
            strs[i] = String.valueOf(preds[i]);
        }
        String content = String.join("\n",strs);
        try{
            Files.write(Paths.get("output.txt"), content.getBytes());
        }catch (IOException e){
            e.printStackTrace();
        }

    }
}
