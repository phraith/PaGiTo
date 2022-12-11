import { HttpTransportType, HubConnection, HubConnectionBuilder, HubConnectionState, LogLevel } from "@microsoft/signalr";

class MessageHubConnectionProvider {
    connection: HubConnection;
    apiKey: string;
    receiveJobResult: (message: string, colormap: string) => void;
    receiveJobInfos: (message: string) => void;
    getLineprofiles: (message: string) => void;
    constructor(
        apiKey: string,
        receiveJobResult: (message: string, colormap: string) => void,
        receiveJobInfos: (message: string) => void,
        getLineprofiles: (message: string) => void) {

        this.apiKey = apiKey;
        this.receiveJobInfos = receiveJobInfos;
        this.receiveJobResult = receiveJobResult;
        this.getLineprofiles = getLineprofiles;

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
                this.connection.on("ReceiveJobId", (message, colormap) => {
                    this.receiveJobResult(message, colormap);
                });

                this.connection.on("ReceiveJobInfos", (message) => {
                    this.receiveJobInfos(message);
                });

                this.connection.on("ProcessLineprofiles", (message) => {
                    this.getLineprofiles(message)
                })
            })
            .catch((e) => console.log("Connection failed: ", e));
    }

    requestProfiles(jsonConfig: any): any {
        if (this.connection?.state === HubConnectionState.Connected) {
            this.connection?.send("GetProfiles", jsonConfig);
        }
    }

    requestJob(jsonConfig: any, colormap: string): any  {
        if (this.connection?.state === HubConnectionState.Connected) {
            console.log("send job")
            this.connection?.send("IssueJob", jsonConfig, colormap);
        }
    }
}

export { MessageHubConnectionProvider }