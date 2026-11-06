Group 4 - Air Pollution Forecasting : CO(GT) predictor
Sai Rohith CHETLAPALLY - Srija POUDAPALLY

HOW TO RUN
1. Install Docker Desktop: https://www.docker.com/products/docker-desktop/
   Open it and wait for "Engine running".
2. Windows: double-click VISITOR_RUN.bat
   Mac / Linux: open a terminal in this folder and run
     docker load -i group4-co-predictor.tar
     docker run -p 8501:8501 -p 8000:8000 -p 5000:5000 group4-co-predictor:1.0
3. Open http://localhost:8501 in your browser.

   App ......... http://localhost:8501   (predict CO from sensor values)
   API docs .... http://localhost:8000/docs
   MLflow ...... http://localhost:5000   (experiments and registered model)

To stop: docker rm -f group4-co
