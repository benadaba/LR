mkdir -p ~/.housingprediction/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.housingprediction/config.toml