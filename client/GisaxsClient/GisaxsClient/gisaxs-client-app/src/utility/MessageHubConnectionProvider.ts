import { HttpTransportType, HubConnection, HubConnectionBuilder, HubConnectionState, LogLevel } from "@microsoft/signalr";

class MessageHubConnectionProvider {
    connection: HubConnection;
    apiKey: string;
    receiveJobResult: (message: string) => void;
    receiveJobInfos: (message: string) => void;
    getLineprofiles: (message: string) => void;
    constructor(
        apiKey: string,
        receiveJobResult: (message: string) => void,
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
                this.connection.on("ReceiveJobId", (message) => {
                    this.receiveJobResult(message);
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
            console.log("after profilessss sent");
        }
    }

    requestJob(jsonConfig: any): any  {
        if (this.connection?.state === HubConnectionState.Connected) {
            this.connection?.send("IssueJob", jsonConfig);
            console.log("after jobbbb sent");
        }
    }
}

export { MessageHubConnectionProvider }