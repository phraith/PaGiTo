#See https://aka.ms/containerfastmode to understand how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM mcr.microsoft.com/dotnet/aspnet:7.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src

COPY . .

WORKDIR "/src/src/Vraith.GisaxsClient"
RUN dotnet build "Vraith.GisaxsClient.csproj" -c Release -o /app/build

FROM build AS publish
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash
RUN apt-get update 
RUN apt-get install -y nodejs
RUN dotnet publish "Vraith.GisaxsClient.csproj" -c Release -o /app/publish

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS tests
COPY . .
RUN  for i in `find . -name "*.Tests.csproj" -type f`; do dotnet add $i package JUnitTestLogger; dotnet add $i package coverlet.collector; done

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "Vraith.GisaxsClient.dll"]
