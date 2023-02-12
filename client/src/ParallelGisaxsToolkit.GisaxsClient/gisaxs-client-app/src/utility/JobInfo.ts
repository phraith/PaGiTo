class JobInfo {
    id: number;
    config: any
    constructor(id: number, info: any)
    {
        this.id = id;
        this.config = info; 
    }
}

export { JobInfo }