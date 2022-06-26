class RelativeLineProfile {
    startRel: Coordinate;
    endRel: Coordinate;

    constructor(start: Coordinate, end: Coordinate, dim: Coordinate) {
        this.startRel = new Coordinate(start.x / dim.x, start.y / dim.y);
        this.endRel = new Coordinate(end.x / dim.x, end.y / dim.y);
    }

    toLineProfile(width: number, height: number): LineProfile
    {
        let start =  new Coordinate(this.startRel.x * width, this.startRel.y * height);
        let end = new Coordinate(this.endRel.x * width, this.endRel.y * height);
        return new LineProfile(start, end)
    }
}

class LineProfile {
    start:  Coordinate;
    end: Coordinate;

    constructor(start: Coordinate, end: Coordinate) {
        this.start = start;
        this.end = end;
    }
}


class Coordinate {
    x: number;
    y: number;

    constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
    }
}

class LineProfileState {
    lineMode: boolean;
    lineProfiles: RelativeLineProfile[];
    currentLineProfile: RelativeLineProfile;
    constructor(lineMode: boolean, lineProfiles: RelativeLineProfile[], currentLineProfile: RelativeLineProfile)
    {
        this.lineMode = lineMode;
        this.lineProfiles = lineProfiles;
        this.currentLineProfile = currentLineProfile;
    }
}


export { RelativeLineProfile, LineProfile, Coordinate, LineProfileState }