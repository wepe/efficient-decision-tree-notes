package test;

import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class tree {
    public bin bining;
    public int blen;
    public tree(bin b){
        bining = b;
        blen = b.a.length;
    }

    class LearnRunnable implements Runnable{
        public int n;
        public LearnRunnable(int i){
            n = i;
        }

        @Override
        public void run(){
            int sum = 0;
            for(int j=0;j<=n;j++){
                sum += Math.pow(2,j);
            }
            bining.a[n] = sum;
        }
    }

    public void learn(){
        ExecutorService pool = Executors.newFixedThreadPool(4);
        for(int i=0;i<blen;i++){
            pool.execute(new LearnRunnable(i));
        }

        pool.shutdown();
        try {
            pool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args){
        bin b = new bin(1000);
        b.display();
        tree t = new tree(b);
        t.learn();
        b.display();

        }


}
