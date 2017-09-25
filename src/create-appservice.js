var AppServiceRegistration = require("matrix-appservice").AppServiceRegistration;

// creating registration files
var reg = new AppServiceRegistration();
reg.setId('polite-appservice');
reg.setAppServiceUrl("http://localhost:8010");
reg.setHomeserverToken(AppServiceRegistration.generateToken());
reg.setAppServiceToken(AppServiceRegistration.generateToken());
reg.setSenderLocalpart("polite-appservice");
reg.addRegexPattern("users", "@.*", true);
reg.setProtocols(["polite"]); // For 3PID lookups
reg.outputAsYaml("registration.yaml");
