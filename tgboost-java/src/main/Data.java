//parse the csv file, get features and label. the format: feature1,feature2,...,label
//first scan, get the feature dimension, dataset size, count of missing value for each feature
//second scan, get each feature's (value,index) and missing value indexes
//if we use ArrayList,only one scanning is needed, but it is memory consumption

package main;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class Data {
    public double[][][] feature_value_index;
    public double[] label;
    public int[][] missing_index;
    public int feature_dim;
    public int dataset_size;
    public ArrayList<Integer> missing_count = new ArrayList<>();
    public Double[][] origin_data;

    public Data(String file){
        first_scan(file);
        second_scan(file);
    }

    //to obtain: feature_dim, dataset_size,missing_count
    private void first_scan(String file){
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            String header = br.readLine();
            feature_dim = header.split(",").length - 1;
            for(int i=0;i<feature_dim;i++){
                missing_count.add(0);
            }

            String line;
            dataset_size = 0;
            while((line = br.readLine()) != null){
                String[] strs = line.split(",");
                dataset_size += 1;
                for(int i=0;i<feature_dim;i++){
                    if(strs[i].equals("")){
                        missing_count.set(i,missing_count.get(i)+1);
                    }
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //to obtain:feature_value_index,label,missing_index,origin_data
    private void second_scan(String file){
        label = new double[dataset_size];
        missing_index = new int[feature_dim][];
        feature_value_index = new double[feature_dim][][];
        origin_data = new Double[dataset_size][feature_dim];

        for(int i=0;i<feature_dim;i++){
            int cnt = missing_count.get(i);
            missing_index[i] = new int[cnt];
            feature_value_index[i] = new double[dataset_size-cnt][2];
        }

        try{
            BufferedReader br = new BufferedReader(new FileReader(file));
            br.readLine();

            int[] cur_index = new int[feature_dim];
            int[] cur_missing_index = new int[feature_dim];
            Arrays.fill(cur_index,0);
            Arrays.fill(cur_missing_index,0);

            for(int row=0;row<dataset_size;row++){
                String[] strs = br.readLine().split(",");
                label[row] = Double.parseDouble(strs[strs.length-1]);

                for(int col=0;col<feature_dim;col++){
                    if(strs[col].equals("")){
                        missing_index[col][cur_missing_index[col]] = row;
                        cur_missing_index[col] += 1;

                        origin_data[row][col] = null;
                    }else{
                        feature_value_index[col][cur_index[col]][0] = Double.parseDouble(strs[col]);
                        feature_value_index[col][cur_index[col]][1] = row;
                        cur_index[col] += 1;

                        origin_data[row][col] = Double.parseDouble(strs[col]);
                    }
                }

            }
        }catch (IOException e){
            e.printStackTrace();
        }
    }
}
