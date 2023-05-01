class JobInfo {
    id: number;
    config: any;
    result: any;
    constructor(id: number, info: any, result: any)
    {
        this.id = id;
        this.config = info;
        this.result = result 
    }
}

export { JobInfo }