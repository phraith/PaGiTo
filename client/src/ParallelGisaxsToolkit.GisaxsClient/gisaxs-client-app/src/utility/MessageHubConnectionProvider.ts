import { HttpTransportType, HubConnection, HubConnectionBuilder, HubConnectionState, LogLevel } from "@microsoft/signalr";

class MessageHubConnectionProvider {
    connection: HubConnection;
    apiKey: string;
    namedMessageHandlers: [string, (message: string) => void][];
    constructor(apiKey: string, namedMessageHandlers: [string, (message: string) => void][]) {

        this.apiKey = apiKey;
        this.namedMessageHandlers = namedMessageHandlers;
        this.connection = new HubConnectionBuilder()
            .withUrl("/message", {
                skipNegotiation: true,
                transport: HttpTransportType.WebSockets,
                accessTokenFactory: () => this.apiKey,
            })
            .configureLogging(LogLevel.Information)
            .withAutomaticReconnect()
            .build()
    }

    connect(): void {
        if (!this.connection) { return; }
        this.connection
            .start()
            .then((result) => {
                console.log("Connected!");
                for (var i = 0; i < this.namedMessageHandlers.length; ++i) {
                    let [name, messageHandler] = this.namedMessageHandlers[i]
                    this.connection.on(name, (message) => {
                        messageHandler(message);
                    });
                }
            })
            .catch((e) => console.log("Connection failed: ", e));
    }
}

export { MessageHubConnectionProvider }