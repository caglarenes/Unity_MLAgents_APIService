using MLAgent.Shared;
using WebApiContrib.Core.Formatter.MessagePack;
using static MLAgent.Shared.MLBrain;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddSingleton<MLBrain>();
//builder.Services.AddMvcCore().AddMessagePackFormatters();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();


app.MapPost("/prediction", (InputClass userInput, MLBrain mlBrain) =>
{
    if (!userInput.IsModelValid())
    {
        return Results.BadRequest();
    }
    
    return Results.Ok(mlBrain.RequestPrediction(userInput));
});

app.Run();
