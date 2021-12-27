mkdir -p ~/.housing_prediction/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.housing_prediction/config.toml.